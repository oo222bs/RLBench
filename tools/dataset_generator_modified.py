from multiprocessing import Process, Manager

from pyrep.const import RenderMode

from rlbench import ObservationConfig
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.backend.utils import task_file_to_task_class
from rlbench.environment import Environment
import rlbench.backend.task as task
from rlbench.demo import Demo
import os
import pickle
from PIL import Image
from rlbench.backend import utils
from rlbench.backend.const import *
import numpy as np
import csv
from absl import app
from absl import flags
from typing import List
import logging

task_list = ['slide_block_to_color_target_simplified']

#task_list = []

#task_list = ['close_jar', 'insert_onto_square_peg', 'light_bulb_in', 'meat_off_grill', 'open_drawer', 'place_cups','place_shape_in_shape_sorter', #'push_buttons', 'put_groceries_in_cupboard', 'put_item_in_drawer', 'put_money_in_safe', 'reach_and_drag']

FLAGS = flags.FLAGS

flags.DEFINE_string('save_path',
                    '../../peract/data/val_slide_block',
                    'Where to save the demos.')
flags.DEFINE_list('tasks', task_list,
                  'The tasks to collect. If empty, all tasks are collected.')
flags.DEFINE_list('image_size', [128, 128],
                  'The size of the images tp save.')
flags.DEFINE_enum('renderer',  'opengl', ['opengl', 'opengl3'],
                  'The renderer to use. opengl does not include shadows, '
                  'but is faster.')
flags.DEFINE_integer('processes', 1,
                     'The number of parallel processes during collection.')
flags.DEFINE_integer('episodes_per_task', 100,
                     'The number of episodes to collect per task.')
flags.DEFINE_integer('variations', -1,
                     'Number of variations to collect per task. -1 for all.')
flags.DEFINE_bool('all_variations', True,
                  'Include all variations when sampling epsiodes')
flags.DEFINE_bool('with_keyframes', True, 'Include keyframes when saving timesteps')
flags.DEFINE_bool('save_joints', True, 'Save joint values for each timestep')
flags.DEFINE_bool('save_ee_pose', True, 'Save the end effector pose for each timestep')

def _is_stopped(demo, i, obs, stopped_buffer, delta=0.1):
    next_is_not_final = i == (len(demo) - 2)
    gripper_state_no_change = (
            i < (len(demo) - 2) and
            (obs.gripper_open == demo[i + 1].gripper_open and
             obs.gripper_open == demo[i - 1].gripper_open and
             demo[i - 2].gripper_open == demo[i - 1].gripper_open))
    small_delta = np.allclose(obs.joint_velocities, 0, atol=delta)
    stopped = (stopped_buffer <= 0 and small_delta and
               (not next_is_not_final) and gripper_state_no_change)
    return stopped


def keypoint_discovery(demo: Demo,
                       stopping_delta=0.1,
                       method='heuristic_and_fixed_interval_frame_before') -> List[int]:
    episode_keypoints = []
    if method == 'heuristic':
        prev_gripper_open = demo[0].gripper_open
        stopped_buffer = 0
        for i, obs in enumerate(demo):
            stopped = _is_stopped(demo, i, obs, stopped_buffer, stopping_delta)
            stopped_buffer = 4 if stopped else stopped_buffer - 1
            # If change in gripper, or end of episode.
            last = i == (len(demo) - 1)
            if i != 0 and (obs.gripper_open != prev_gripper_open or
                           last or stopped):
                episode_keypoints.append(i)
            prev_gripper_open = obs.gripper_open
        if len(episode_keypoints) > 1 and (episode_keypoints[-1] - 1) == \
                episode_keypoints[-2]:
            episode_keypoints.pop(-2)
        logging.debug('Found %d keypoints.' % len(episode_keypoints),
                      episode_keypoints)
        return episode_keypoints

    elif method == 'random':
        # Randomly select keypoints.
        episode_keypoints = np.random.choice(
            range(len(demo)),
            size=20,
            replace=False)
        episode_keypoints.sort()
        return episode_keypoints

    elif method == 'fixed_interval':
        # Fixed interval.
        episode_keypoints = []
        segment_length = len(demo) // 20
        for i in range(0, len(demo), segment_length):
            episode_keypoints.append(i)
        return episode_keypoints

    elif method == 'heuristic_and_fixed_interval':
        prev_gripper_open = demo[0].gripper_open
        stopped_buffer = 0
        for i, obs in enumerate(demo):
            stopped = _is_stopped(demo, i, obs, stopped_buffer, stopping_delta)
            stopped_buffer = 4 if stopped else stopped_buffer - 1
            # If change in gripper, or end of episode.
            last = i == (len(demo) - 1)
            interval = i % 10 == 0
            if obs.gripper_open != prev_gripper_open or last or stopped or interval:
                episode_keypoints.append(i)
            prev_gripper_open = obs.gripper_open
        if len(episode_keypoints) > 1 and (episode_keypoints[-1] - 1) == \
                episode_keypoints[-2]:
            episode_keypoints.pop(-2)
        logging.debug('Found %d keypoints.' % len(episode_keypoints),
                      episode_keypoints)
        return episode_keypoints
    elif method == 'heuristic_and_fixed_interval_frame_before':
        prev_gripper_open = demo[0].gripper_open
        stopped_buffer = 0
        for i, obs in enumerate(demo):
            stopped = _is_stopped(demo, i, obs, stopped_buffer, stopping_delta)
            stopped_buffer = 4 if stopped else stopped_buffer - 1
            # If change in gripper, or end of episode.
            last = i == (len(demo) - 1)
            interval = i % 5 == 0
            if obs.gripper_open != prev_gripper_open or last or stopped:
                episode_keypoints.append(i)
                if i>0 and (i-1) not in episode_keypoints:
                    episode_keypoints.append(i-1)
            elif interval:
                episode_keypoints.append(i)
            prev_gripper_open = obs.gripper_open
        episode_keypoints.sort()
        if len(episode_keypoints) > 1 and (episode_keypoints[-1] - 1) == \
                episode_keypoints[-2]:
            episode_keypoints.pop(-2)
        logging.debug('Found %d keypoints.' % len(episode_keypoints),
                      episode_keypoints)
        return episode_keypoints
    else:
        raise NotImplementedError


def check_and_make(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def save_demo(demo, example_path, variation):

    # Save image data first, and then None the image data, and pickle
    left_shoulder_rgb_path = os.path.join(
         example_path, LEFT_SHOULDER_RGB_FOLDER)
    right_shoulder_rgb_path = os.path.join(
        example_path, RIGHT_SHOULDER_RGB_FOLDER)
    wrist_rgb_path = os.path.join(example_path, WRIST_RGB_FOLDER)
    front_rgb_path = os.path.join(example_path, FRONT_RGB_FOLDER)
    wrist_depth_path = os.path.join(example_path, WRIST_DEPTH_FOLDER)

    check_and_make(left_shoulder_rgb_path)
    check_and_make(right_shoulder_rgb_path)
    check_and_make(wrist_rgb_path)
    check_and_make(front_rgb_path)
    check_and_make(wrist_depth_path)

    filename = os.path.join(example_path) + "/joint_seq.csv"
    filename_pose  = os.path.join(example_path) + "/pose_seq.csv"
    prop_log = []
    pose_log = []
    for i, obs in enumerate(demo):
        #if i % 10 == 0 or i + 1 == len(demo):
        if obs.left_shoulder_rgb is None:#len(obs.left_shoulder_rgb) == 0:
            break
        left_shoulder_rgb = Image.fromarray(obs.left_shoulder_rgb)
        right_shoulder_rgb = Image.fromarray(obs.right_shoulder_rgb)
        wrist_rgb = Image.fromarray(obs.wrist_rgb)
        front_rgb = Image.fromarray(obs.front_rgb)
        wrist_depth = utils.float_array_to_rgb_image(obs.wrist_depth, scale_factor=DEPTH_SCALE)
	
        left_shoulder_rgb.save(
            os.path.join(left_shoulder_rgb_path, IMAGE_FORMAT % i))
        right_shoulder_rgb.save(
            os.path.join(right_shoulder_rgb_path, IMAGE_FORMAT % i))
        wrist_rgb.save(os.path.join(wrist_rgb_path, IMAGE_FORMAT % i))
        front_rgb.save(os.path.join(front_rgb_path, IMAGE_FORMAT % i))
        wrist_depth.save(os.path.join(wrist_depth_path, IMAGE_FORMAT % i))
        if FLAGS.save_joints:
            joints = obs.joint_positions
            gripper = obs.gripper_open
            proprioception = np.append(joints, [gripper], axis=-1)
            prop_log.append(proprioception)
        if FLAGS.save_ee_pose:
            pose = obs.gripper_pose
            gripper = obs.gripper_open
            ic = obs.ignore_collisions
            proprioception = np.append(pose, [gripper, ic], axis=-1)
            pose_log.append(proprioception)
        # We save the images separately, so set these to None for pickling.
        obs.left_shoulder_rgb = None
        obs.left_shoulder_depth = None
        obs.left_shoulder_point_cloud = None
        obs.left_shoulder_mask = None
        obs.right_shoulder_rgb = None
        obs.right_shoulder_depth = None
        obs.right_shoulder_point_cloud = None
        obs.right_shoulder_mask = None
        obs.overhead_rgb = None
        obs.overhead_depth = None
        obs.overhead_point_cloud = None
        obs.overhead_mask = None
        obs.wrist_rgb = None
        obs.wrist_depth = None
        obs.wrist_point_cloud = None
        obs.wrist_mask = None
        obs.front_rgb = None
        obs.front_depth = None
        obs.front_point_cloud = None
        obs.front_mask = None

    # Save the joint values and gripper open
    with open(filename, "w") as csvfile:
        filewriter = csv.writer(csvfile, delimiter=",", quotechar="|")
        filewriter.writerows(prop_log)
    prop_log = []

    # Save poses and gripper open
    with open(filename_pose, "w") as csvfile:
        filewriter = csv.writer(csvfile, delimiter=",", quotechar="|")
        filewriter.writerows(pose_log)
    pose_log = []

    # Save the low-dimension data
    with open(os.path.join(example_path, LOW_DIM_PICKLE), 'wb') as f:
        pickle.dump(demo, f)

    with open(os.path.join(example_path, VARIATION_NUMBER), 'wb') as f:
        pickle.dump(variation, f)


def run(i, lock, task_index, variation_count, results, file_lock, tasks):
    """Each thread will choose one task and variation, and then gather
    all the episodes_per_task for that variation."""

    # Initialise each thread with random seed
    np.random.seed(None)
    num_tasks = len(tasks)

    img_size = list(map(int, FLAGS.image_size))

    obs_config = ObservationConfig()
    obs_config.set_all(True)
    obs_config.right_shoulder_camera.image_size = img_size
    obs_config.left_shoulder_camera.image_size = img_size
    obs_config.overhead_camera.image_size = img_size
    obs_config.wrist_camera.image_size = img_size
    obs_config.front_camera.image_size = img_size

    # Store depth as 0 - 1
    obs_config.right_shoulder_camera.depth_in_meters = False
    obs_config.left_shoulder_camera.depth_in_meters = False
    obs_config.overhead_camera.depth_in_meters = False
    obs_config.wrist_camera.depth_in_meters = False
    obs_config.front_camera.depth_in_meters = False

    # We want to save the masks as rgb encodings.
    obs_config.left_shoulder_camera.masks_as_one_channel = False
    obs_config.right_shoulder_camera.masks_as_one_channel = False
    obs_config.overhead_camera.masks_as_one_channel = False
    obs_config.wrist_camera.masks_as_one_channel = False
    obs_config.front_camera.masks_as_one_channel = False

    if FLAGS.renderer == 'opengl':
        obs_config.right_shoulder_camera.render_mode = RenderMode.OPENGL
        obs_config.left_shoulder_camera.render_mode = RenderMode.OPENGL
        obs_config.overhead_camera.render_mode = RenderMode.OPENGL
        obs_config.wrist_camera.render_mode = RenderMode.OPENGL
        obs_config.front_camera.render_mode = RenderMode.OPENGL

    rlbench_env = Environment(
        action_mode=MoveArmThenGripper(JointVelocity(), Discrete()),
        obs_config=obs_config,
        headless=True)
    rlbench_env.launch()

    task_env = None

    tasks_with_problems = results[i] = ''

    while True:
        # Figure out what task/variation this thread is going to do
        with lock:

            if task_index.value >= num_tasks:
                print('Process', i, 'finished')
                break

            my_variation_count = variation_count.value
            t = tasks[task_index.value]
            task_env = rlbench_env.get_task(t)
            var_target = task_env.variation_count()
            if FLAGS.variations >= 0:
                var_target = np.minimum(FLAGS.variations, var_target)
            if my_variation_count >= var_target:
                # If we have reached the required number of variations for this
                # task, then move on to the next task.
                variation_count.value = my_variation_count = 0
                task_index.value += 1

            variation_count.value += 1
            if task_index.value >= num_tasks:
                print('Process', i, 'finished')
                break
            t = tasks[task_index.value]

        variation_path = os.path.join(
            FLAGS.save_path, task_env.get_name(),
            VARIATIONS_FOLDER % my_variation_count)
        check_and_make(variation_path)

        episodes_path = os.path.join(variation_path, EPISODES_FOLDER)
        check_and_make(episodes_path)

        abort_variation = False
        for ex_idx in range(FLAGS.episodes_per_task):
            print('Process', i, '// Task:', task_env.get_name(),
                  '// Variation:', my_variation_count, '// Demo:', ex_idx)
            attempts = 10
            while attempts > 0:
                try:
                    task_env = rlbench_env.get_task(t)
                    task_env.set_variation(my_variation_count)
                    descriptions, obs = task_env.reset()

                    # TODO: for now we do the explicit looping.
                    demo, = task_env.get_demos(
                        amount=1,
                        live_demos=True)
                except Exception as e:
                    attempts -= 1
                    if attempts > 0:
                        continue
                    problem = (
                        'Process %d failed collecting task %s (variation: %d, '
                        'example: %d). Skipping this task/variation.\n%s\n' % (
                            i, task_env.get_name(), my_variation_count, ex_idx,
                            str(e))
                    )
                    print(problem)
                    tasks_with_problems += problem
                    abort_variation = True
                    break
                episode_path = os.path.join(episodes_path, EPISODE_FOLDER % ex_idx)
                with file_lock:
                    save_demo(demo, episode_path, my_variation_count)

                    with open(os.path.join(
                            episode_path, VARIATION_DESCRIPTIONS), 'wb') as f:
                        pickle.dump(descriptions, f)
                break
            if abort_variation:
                break

    results[i] = tasks_with_problems
    rlbench_env.shutdown()


def run_all_variations(i, lock, task_index, variation_count, results, file_lock, tasks):
    """Each thread will choose one task and variation, and then gather
    all the episodes_per_task for that variation."""

    # Initialise each thread with random seed
    np.random.seed(None)
    num_tasks = len(tasks)

    img_size = list(map(int, FLAGS.image_size))

    obs_config = ObservationConfig()
    obs_config.set_all(True)
    obs_config.right_shoulder_camera.image_size = img_size
    obs_config.left_shoulder_camera.image_size = img_size
    obs_config.overhead_camera.image_size = img_size
    obs_config.wrist_camera.image_size = img_size
    obs_config.front_camera.image_size = img_size

    # Store depth as 0 - 1
    obs_config.right_shoulder_camera.depth_in_meters = False
    obs_config.left_shoulder_camera.depth_in_meters = False
    obs_config.overhead_camera.depth_in_meters = False
    obs_config.wrist_camera.depth_in_meters = False
    obs_config.front_camera.depth_in_meters = False

    # We want to save the masks as rgb encodings.
    obs_config.left_shoulder_camera.masks_as_one_channel = False
    obs_config.right_shoulder_camera.masks_as_one_channel = False
    obs_config.overhead_camera.masks_as_one_channel = False
    obs_config.wrist_camera.masks_as_one_channel = False
    obs_config.front_camera.masks_as_one_channel = False

    if FLAGS.renderer == 'opengl':
        obs_config.right_shoulder_camera.render_mode = RenderMode.OPENGL
        obs_config.left_shoulder_camera.render_mode = RenderMode.OPENGL
        obs_config.overhead_camera.render_mode = RenderMode.OPENGL
        obs_config.wrist_camera.render_mode = RenderMode.OPENGL
        obs_config.front_camera.render_mode = RenderMode.OPENGL

    rlbench_env = Environment(
        action_mode=MoveArmThenGripper(JointVelocity(), Discrete()),
        obs_config=obs_config,
        headless=True)
    rlbench_env.launch()

    task_env = None
    offset=0
    tasks_with_problems = results[i] = ''
    while True:
        # with lock:
        if task_index.value >= num_tasks:
            print('Process', i, 'finished')
            break

        t = tasks[task_index.value]

        task_env = rlbench_env.get_task(t)
        possible_variations = task_env.variation_count()

        variation_path = os.path.join(
            FLAGS.save_path, task_env.get_name(),
            VARIATIONS_ALL_FOLDER)
        check_and_make(variation_path)

        episodes_path = os.path.join(variation_path, EPISODES_FOLDER)
        check_and_make(episodes_path)

        abort_variation = False
        for ex_idx in range(FLAGS.episodes_per_task-offset):
            attempts = 10
            while attempts > 0:
                try:
                    variation = np.random.randint(possible_variations)
                    task_env = rlbench_env.get_task(t)
                    task_env.set_variation(variation)
                    descriptions, obs = task_env.reset()

                    print('Process', i, '// Task:', task_env.get_name(),
                          '// Variation:', variation, '// Demo:', (ex_idx+offset))

                    # TODO: for now we do the explicit looping.
                    demo, = task_env.get_demos(
                        amount=1,
                        live_demos=True)
                except Exception as e:
                    attempts -= 1
                    if attempts > 0:
                        continue
                    problem = (
                        'Process %d failed collecting task %s (variation: %d, '
                        'example: %d). Skipping this task/variation.\n%s\n' % (
                            i, task_env.get_name(), variation, (ex_idx+offset),
                            str(e))
                    )
                    print(problem)
                    tasks_with_problems += problem
                    abort_variation = True
                    break
                episode_path = os.path.join(episodes_path, EPISODE_FOLDER % (ex_idx+offset))
                # take every tenth observation plus the last or also keyframes
                #if FLAGS.with_keyframes:
                #    keypoint_tsteps = keypoint_discovery(demo)
                #    demo = [demo[ts] for ts in keypoint_tsteps]
                #else:
                #    last_obs = demo[-1]
                #    demo = demo[::10]
                #    demo.append(last_obs)
                with file_lock:
                    save_demo(demo, episode_path, variation)

                    with open(os.path.join(
                            episode_path, VARIATION_DESCRIPTIONS), 'wb') as f:
                        pickle.dump(descriptions, f)
                break
            if abort_variation:
                break

        # with lock:
        task_index.value += 1

    results[i] = tasks_with_problems
    rlbench_env.shutdown()


def main(argv):

    task_files = [t.replace('.py', '') for t in os.listdir(task.TASKS_PATH)
                  if t != '__init__.py' and t.endswith('.py')]

    if len(FLAGS.tasks) > 0:
        for t in FLAGS.tasks:
            if t not in task_files:
                raise ValueError('Task %s not recognised!.' % t)
        task_files = FLAGS.tasks

    tasks = [task_file_to_task_class(t) for t in task_files]

    manager = Manager()

    result_dict = manager.dict()
    file_lock = manager.Lock()

    task_index = manager.Value('i', 0)
    variation_count = manager.Value('i', 0)
    lock = manager.Lock()

    check_and_make(FLAGS.save_path)

    if FLAGS.all_variations:
        # multiprocessing for all_variations not support (for now)
        run_all_variations(0, lock, task_index, variation_count, result_dict, file_lock, tasks)
    else:
        processes = [Process(
            target=run, args=(
                i, lock, task_index, variation_count, result_dict, file_lock,
                tasks))
            for i in range(FLAGS.processes)]
        [t.start() for t in processes]
        [t.join() for t in processes]

    print('Data collection done!')
    for i in range(FLAGS.processes):
        print(result_dict[i])


if __name__ == '__main__':
  app.run(main)

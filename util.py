from skimage import transform # Help us to preprocess the frames
from skimage.color import rgb2gray # Help us to gray our frames
import numpy as np
from collections import deque# Ordered collection with ends
import cv2

def preprocess_frame(frame, decay_step = 0):
    # Greyscale frame
    gray = rgb2gray(frame)
    # gray = frame

    # Crop the screen (remove the part below the player)
    # [Up: Down, Left: right]
    cropped_frame = gray[15:-19,15:-15]
    # 190x210

    # Normalize Pixel Values
    normalized_frame = cropped_frame/255.0

    # if decay_step > 15:
    # cv2.imwrite("frame.jpg", frame)
    # cv2.imwrite("gray.jpg", gray)
    # cv2.imwrite("cropped_frame.jpg", cropped_frame)
    # cv2.imwrite("resized.jpg", transform.resize(normalized_frame, [38,42]))

    # Resize
    # Thanks to Mikołaj Walkowiak
    # preprocessed_frame = transform.resize(normalized_frame, [224,240])
    # preprocessed_frame = transform.resize(normalized_frame, [95, 105])
    # preprocessed_frame = transform.resize(normalized_frame, [76, 84])
    # preprocessed_frame = transform.resize(normalized_frame, [54, 60])
    # 76, 84
    preprocessed_frame = transform.resize(normalized_frame, [38, 42])

    return preprocessed_frame # 110x84x1 frame

def stack_frames(stacked_frames, state, is_new_episode, decay_step = 0):
    # Preprocess frame
    frame = preprocess_frame(state, decay_step)

    if is_new_episode:
        # Clear our stacked_frames
        # stacked_frames  =  deque([np.zeros((224, 240), dtype=np.int) for i in range(4)], maxlen=4)
        stacked_frames = deque([np.zeros((38, 42), dtype=np.int) for i in range(4)], maxlen=4)

        # Because we're in a new episode, copy the same frame 4x
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)

        # Stack the frames
        stacked_state = np.stack(stacked_frames, axis=2)

    else:
        # Append frame to deque, automatically removes the oldest frame
        stacked_frames.append(frame)

        # Build the stacked state (first dimension specifies different frames)
        stacked_state = np.stack(stacked_frames, axis=2)

    return stacked_state, stacked_frames

def predict_action(explore_start, explore_stop, decay_rate, decay_step, state, actions):
    ## EPSILON GREEDY STRATEGY
    # Choose action a from state s using epsilon greedy.
    ## First we randomize a number
    exp_exp_tradeoff = np.random.rand()

    # Here we'll use an improved version of our epsilon greedy strategy used in Q-learning notebook
    explore_probability = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * decay_step)

    if (explore_probability > exp_exp_tradeoff):
        # Make a random action (exploration)
        choice = random.randint(1,len(possible_actions))-1
        action = possible_actions[choice]

    else:
        # Get action from Q-network (exploitation)
        # Estimate the Qs values state
        Qs = sess.run(DQNetwork.output, feed_dict = {DQNetwork.inputs_: state.reshape((1, *state.shape))})

        # Take the biggest Q value (= the best action)
        choice = np.argmax(Qs)
        action = possible_actions[choice]

    return action, explore_probability
import gym
import numpy as np
import torch
from mujoco_py import GlfwContext
from tqdm import tqdm
import argparse

# from transformers import DecisionTransformerModel
from decision_transformer.models.decision_transformer import DecisionTransformer


GlfwContext(offscreen=True)  # Create a window to init GLFW.

# build the environment
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Simulation parameters")
    parser.add_argument('--env', type=str, default='Hopper-v3', help='Gym environment name')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run the model on')
    parser.add_argument('--model_path', type=str, help='Path to the model file')

    args = parser.parse_args()

    print(f"Environment: {args.env}")
    print(f"Device: {args.device}")
    print(f"Model Path: {args.model_path}")

    env = gym.make(args.env)
    state_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    max_ep_len = 1000
    device = args.device
    scale = 1000.0  # normalization for rewards/returns
    TARGET_RETURN = 3600 / scale  # evaluation conditioning targets, 3600 is reasonable from the paper LINK
    if args.env == 'Hopper-v3':
        state_mean = np.array([1.311279, -0.08469521, -0.5382719, -0.07201576, 0.04932366, 2.1066856,
                               -0.15017354, 0.00878345, -0.2848186,-0.18540096,-0.28461286,])
        state_std = np.array([0.17790751, 0.05444621, 0.21297139, 0.14530419, 0.6124444, 0.85174465,
                              1.4515252, 0.6751696, 1.536239, 1.6160746, 5.6072536])
    elif args.env == 'HalfCheetah-v3':
        state_mean = np.array([-0.06845774, 0.01641455, -0.18354906, -0.27624607, -0.34061527, -0.09339716,
                               -0.21321271, -0.08774239, 5.1730075, -0.04275195, -0.03610836, 0.14053793,
                               0.06049833, 0.09550975, 0.067391, 0.00562739, 0.01338279])
        state_std = np.array([0.07472999, 0.30234998, 0.3020731, 0.34417078, 0.17619242, 0.5072056, 0.25670078,
                              0.32948127, 1.2574149,  0.7600542,  1.9800916, 6.5653625, 7.4663677, 4.472223,
                              10.566964, 5.6719327, 7.498259])
    elif args.env == 'Walker2d-v3':
        state_mean = np.array([1.218966, 0.14163373, -0.03704914, -0.1381431, 0.51382244, -0.0471911,
                               -0.47288352, 0.04225416, 2.3948874, -0.03143199, 0.04466356, -0.02390724,
                               -0.10134014, 0.09090938, -0.00419264, -0.12120572, -0.5497064])
        state_std = np.array([0.12311358, 0.324188, 0.11456084, 0.26230657, 0.5640279, 0.22718786,
                              0.38373196, 0.7373677, 1.2387927, 0.7980206, 1.5664079, 1.8092705,
                              3.0256042, 4.062486, 1.4586568, 3.744569, 5.585129])
    else:
        raise ValueError(f"Unsupported environment: {args.env}. Please select one from: Hopper-v3, HalfCheetah-v3, Walker2d-v3")
    state_mean = torch.from_numpy(state_mean).to(device=device)
    state_std = torch.from_numpy(state_std).to(device=device)

    model_path = args.model_path
    state_dict = torch.load(model_path, map_location=torch.device(device))
    model = DecisionTransformer(
        state_dim=state_dim,
        act_dim=act_dim,
        max_length=20, # K
        max_ep_len=max_ep_len,
        hidden_size=128, # embed_dim
        n_layer=3, # n_layer
        n_head=1, # n_head
        n_inner=4*128, # 4*embed_dim
        activation_function='relu', # activation_function
        n_positions=1024,
        resid_pdrop=0.1, # dropout
        attn_pdrop=0.1, # dropout
    )
    model.load_state_dict(state_dict)  # Load the state dictionary into the model
    model.to(torch.device('cuda'))  # Move the model to GPU
    model.eval()  # Set the model to evaluation mode

    for ep in range(10):
        episode_return, episode_length = 0, 0
        state = env.reset()
        target_return = torch.tensor(TARGET_RETURN, device=device, dtype=torch.float32).reshape(1, 1)
        states = torch.from_numpy(state).reshape(1, state_dim).to(device=device, dtype=torch.float32)
        actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
        rewards = torch.zeros(0, device=device, dtype=torch.float32)

        timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)
        for t in tqdm(range(max_ep_len), desc="Simulation Progress"):
            env.render()
            # add padding
            actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)
            rewards = torch.cat([rewards, torch.zeros(1, device=device)])

            action = model.get_action(
                # model,
                (states.to(dtype=torch.float32) - state_mean) / state_std,
                actions.to(dtype=torch.float32),
                rewards.to(dtype=torch.float32),
                target_return.to(dtype=torch.float32),
                timesteps.to(dtype=torch.long),
            )
            actions[-1] = action
            action = action.detach().cpu().numpy()

            state, reward, done, _ = env.step(action)

            cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)
            states = torch.cat([states, cur_state], dim=0)
            rewards[-1] = reward

            pred_return = target_return[0, -1] - (reward / scale)
            target_return = torch.cat([target_return, pred_return.reshape(1, 1)], dim=1)
            timesteps = torch.cat([timesteps, torch.ones((1, 1), device=device, dtype=torch.long) * (t + 1)], dim=1)

            episode_return += reward
            episode_length += 1

            if done:
                break
    
    env.close()

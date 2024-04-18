import torch
from llm2vec import LLM2Vec
import json

from transformers import AutoTokenizer, AutoModel, AutoConfig
from peft import PeftModel

access_token = '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'

ENV_DESCRIPTION = "Clean up is a public goods dilemma in which agents get a reward for consuming apples, but must use a cleaning beam to clean a river in order for apples to grow. While an agent is cleaning the river, other agents can exploit it by consuming the apples that appear."
INSTRUCTION = "The environment is a 7 x 7 grid which comprises water and apples and 2 agents. Environment is indexed from 0-6. Apples can grow in positions [[5,0],[5,1],[5,2],[5,3],[5,4],[5,5],[5,6],[6,0],[6,1],[6,2],[6,3],[6,4],[6,5],[6,6]]. Water is in [[0,0],[0,1],[0,2],[1,0],[1,1],[2,0],[2,1],[3,0]]. The agents can move in the 4 cardinal directions. The agents can consume apples and clean the river. The agents must learn to cooperate to maximize their reward."
INSTRUCTION += "Your taks is to encode the given trajectory that is obtained from a policy model for each agent. The trajectory is in the format : (Step No.) - Agent 1 (x,y,orientation(NSEW),action,reward) - Agent 2 (x,y,orientation,action,reward) - Clean Water[(x,y)] - Unclean Water [(x,y)] - Apples[(x,y)]. Note that Clean Water, Unclean Water and Apples are a list of coordinates. The trajectory is a list of such steps."


def get_water_coords(observations):
    water_coords = [[0,0],[0,1],[0,2],[1,0],[1,1],[2,0],[2,1],[3,0]]
    clean_list = []
    unclean_list = []
    for j, clean in enumerate(observations):
        if clean:
            clean_list.append(water_coords[j])
        else:
            unclean_list.append(water_coords[j])
    return clean_list, unclean_list

def get_apple_coords(observations):
    growable_coords = [[5,0],[5,1],[5,2],[5,3],[5,4],[5,5],[5,6],[6,0],[6,1],[6,2],[6,3],[6,4],[6,5],[6,6]]
    apple_list = []
    for j, apple in enumerate(observations):
        if apple:
            apple_list.append(growable_coords[j])
    return apple_list

def convert_orientation_to_text(orientation):
    # print(orientation)
    if orientation[0]:
        return "North"
    elif orientation[2]:
        return "South"
    elif orientation[1]:
        return "East"
    elif orientation[3]:
        return "West"
    else:
        raise ValueError("Invalid orientation value")
    
def convert_transition_to_text(transition):
    actions = transition['action']
    observations = transition['observation'][0]
    agent1_pos = observations[0:2]
    agent1_orientation = convert_orientation_to_text(observations[2:6])
    agent2_pos = observations[6:8]
    agent2_orientation = convert_orientation_to_text(observations[8:12])
    
    clean_water, unclean_water = get_water_coords(observations[12:20])
    apple_list = get_apple_coords(observations[20:34])
    rewards = transition['reward']
    # print(actions)
    # print(rewards)
    
    text = "Agent 1 ({} ,{}, {}, {}, {} ".format(agent1_pos[0], agent1_pos[1], agent1_orientation, actions[0], rewards[0])
    text += " - Agent 2 ({} ,{}, {}, {}, {} )".format(agent2_pos[0], agent2_pos[1], agent2_orientation, actions[1], rewards[1])
    text += " - Clean Water: " + str(clean_water)
    text += " - Unclean Water: " + str(unclean_water)
    text += " - Apples: " + str(apple_list) + " | "
    return text
    
def convert_traj_to_text(traj):
    text = ""
    for i, transition in enumerate(traj):
        text += "Step" + str(i) + " - " + convert_transition_to_text(transition)
        
    return text

def get_full_trajectory_text(traj):
    text = convert_traj_to_text(traj)
    return ENV_DESCRIPTION + "." + INSTRUCTION + "." + text + "."

def read_trajectory(filename):
    with open(filename, 'r') as file:
        data = json.load(file)
    return data

def embed_trajectory(model, traj):
    text = get_full_trajectory_text(traj)
    print("Length of Text: {}".format(len(text)))
    return model.encode(text)

def init_model(model_type):
    # model = LLM2Vec.from_pretrained(
    #     model_type,
    #     peft_model_name_or_path="McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp-supervised",
    #     device_map="cuda" if torch.cuda.is_available() else "cpu",
    #     torch_dtype=torch.float32,
    # )
    
    # Loading base Mistral model, along with custom code that enables bidirectional connections in decoder-only LLMs. MNTP LoRA weights are merged into the base model.
    with torch.no_grad():
        tokenizer = AutoTokenizer.from_pretrained(
            "McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp",
            token = access_token
        )
        config = AutoConfig.from_pretrained(
            "McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp", trust_remote_code=True,
            token = access_token
        )
        model = AutoModel.from_pretrained(
            "McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp",
            trust_remote_code=True,
            config=config,
            torch_dtype=torch.bfloat16,
            token = access_token
        )
        model = PeftModel.from_pretrained(
            model,
            "McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp",
            token = access_token
        )
        model = model.merge_and_unload()  # This can take several minutes on cpu

        # # Loading supervised model. This loads the trained LoRA weights on top of MNTP model. Hence the final weights are -- Base model + MNTP (LoRA) + supervised (LoRA).
        # model = PeftModel.from_pretrained(
        #     model, "McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp-supervised",
        #     token = access_token
        #)

    # Wrapper for encoding and pooling operations
    l2v = LLM2Vec(model, tokenizer, pooling_mode="mean", max_length=512)
    return l2v

model = init_model("McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp")
traj00 = read_trajectory('Traj/optimal/0_0_trajectory.json')
traj01 = read_trajectory('Traj/optimal/0_1_trajectory.json')
traj10 = read_trajectory('Traj/optimal/1_0_trajectory.json')
traj11 = read_trajectory('Traj/optimal/1_1_trajectory.json')

traj00_embedding = embed_trajectory(model, traj00)
traj01_embedding = embed_trajectory(model, traj01)
traj10_embedding = embed_trajectory(model, traj10)
traj11_embedding = embed_trajectory(model, traj11)

matrix = torch.stack([traj00_embedding, traj01_embedding, traj10_embedding, traj11_embedding])
similarity = matrix @ matrix.T
print(similarity.cpu().numpy())
from transformers import RobertaTokenizer, RobertaModel
import torch
import torch.nn as nn
import pickle
import torch.optim as optim
import random
import sys
import datetime

class RobertaLinearClassifier(nn.Module):
    def __init__(self, input_len, num_labels):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained('roberta-base', add_pooling_layer=False, output_attentions=False, output_hidden_states=False)
        self.classifier = nn.Linear(768, num_labels) # pretrained Roberta has 768 hidden layers; classifier produces a single classification value
    
    def forward(self, x, attntn_msk, return_features=False):
        features = self.roberta(x, attention_mask=attntn_msk)[0]
        x = features[:, 0, :]
        x = self.classifier(x)
        
        if return_features:
            return x, features[:, 0, :]
        else:
            return x

BATCH_SIZE = 64
NUM_EPOCHS = 1
STOP_AFTER_ONE_BATCH = False

LOAD_FROM_FILE = False
FILE_NAME = "model-10-21-38-32-bios.pkl"

MAX_TRAIN_SIZE = 1000
MAX_TEST_SIZE = 750

FILES = ['bios.pkl', 'bios.pkl', 'subsampled_bios.pkl', 'supersampled_bios.pkl', 'counter_fact_bios.pkl']
DATA_TYPES = ['raw', 'bio', 'raw', 'raw', 'raw', 'raw']

def main():
    for i in range(len(FILES)):
        print(FILES[i], DATA_TYPES[i])
        TRAINING_PERCENT = 0.90
        bios_file = open(FILES[i], 'rb')
        bios = pickle.load(bios_file)
        inputs = [data[DATA_TYPES[i]] for data in bios if len(data[DATA_TYPES[i]]) <= 512]

        tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        longest_input_len = 512
        tokenized_inputs = tokenizer(inputs, padding='max_length', max_length=longest_input_len)

        intended_outputs = [data['title'] for data in bios]
        unique_outputs = list(set(intended_outputs))
        output_map = {unique_outputs[i]:i for i in range(len(unique_outputs))}
        intended_outputs_n = [output_map[output] for output in intended_outputs]

        input_output_pairs = [{'input': tokenized_inputs['input_ids'][i], 'attntn_msk': tokenized_inputs['attention_mask'][i], 'output': intended_outputs_n[i], 'gender': bios[i]['gender']} for i in range(len(inputs))]
        iop_train = [entry for (i, entry) in enumerate(input_output_pairs) if i <= TRAINING_PERCENT * len(input_output_pairs) and i < MAX_TRAIN_SIZE]
        iop_test = [entry for (i, entry) in enumerate(input_output_pairs) if i > TRAINING_PERCENT * len(input_output_pairs) and i >= len(input_output_pairs) - MAX_TEST_SIZE] # include only last MAX_TEST_SIZE items

        print('%d total points; %d train points; %d test points' % (len(inputs), len(iop_train), len(iop_test)))

        rlc = None
        if not LOAD_FROM_FILE:
            rlc = train_model(iop_train, longest_input_len, unique_outputs, i)
        else:
            rlc = RobertaLinearClassifier(longest_input_len, len(unique_outputs))
            rlc.load_state_dict(torch.load(FILE_NAME))
            rlc.eval()
        
        res_file = FILES[i][0:FILES[i].index('.')] + DATA_TYPES[i] + 'results.txt'
        test_model(rlc, iop_test, res_file, unique_outputs)

def train_model(iop, longest_input_len, unique_outputs, i):
    rlc = RobertaLinearClassifier(longest_input_len, len(unique_outputs))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(rlc.parameters(), lr=0.001, momentum=0.9)

    for epoch_idx in range(NUM_EPOCHS):  # loop over the dataset multiple times 
        random.shuffle(iop)

        for batch_idx in range(0, len(iop), BATCH_SIZE):
            batch = iop[batch_idx:batch_idx+BATCH_SIZE]

            selected_input_t = torch.LongTensor([sample['input'] for sample in batch])
            attntn_msk_t = torch.FloatTensor([sample['attntn_msk'] for sample in batch])
            intended_outputs_t = torch.LongTensor([sample['output'] for sample in batch])

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = rlc(selected_input_t, attntn_msk=attntn_msk_t)
            loss = criterion(outputs, intended_outputs_t)
            loss.backward()
            optimizer.step()
            
            print("epoch ", epoch_idx, "batch ", batch_idx // BATCH_SIZE, loss.item())
    
            if STOP_AFTER_ONE_BATCH:
                break

    file_name = 'model-%s-%s' % (datetime.datetime.now().strftime("%d-%H-%M-%S"), FILES[i])
    torch.save(rlc.state_dict(), file_name)
    print('saved to %s' % file_name)

    return rlc

def test_model(rlc, iop, file_name, unique_outputs):
    correct_count = 0
    female_bias_titles = {'yoga_teacher': 0, 'personal_trainer': 0, 'model': 0, 'teacher': 0, 'professor': 0, 'software_engineer': 0, 'attorney': 0, 'physician': 0, 'nurse': 0}
    male_bias_titles = {'yoga_teacher': 0, 'personal_trainer': 0, 'model': 0, 'teacher': 0, 'professor': 0, 'software_engineer': 0, 'attorney': 0, 'physician': 0, 'nurse': 0}
    female_incorrect = 0
    male_incorrect = 0
    female_incorrect_on = {'composer': 0, 'poet': 0, 'psychologist': 0, 'attorney': 0, 'filmmaker': 0, 'software_engineer': 0, 'personal_trainer': 0, 'rapper': 0, 'photographer': 0, 'surgeon': 0, 'accountant': 0, 'architect': 0, 'physician': 0, 'comedian': 0, 'dietitian': 0, 'paralegal': 0, 'dj': 0, 'teacher': 0, 'nurse': 0, 'professor': 0, 'pastor': 0, 'interior_designer': 0, 'yoga_teacher': 0, 'dentist': 0, 'model': 0, 'journalist': 0, 'chiropractor': 0, 'painter': 0}
    male_incorrect_on = {'composer': 0, 'poet': 0, 'psychologist': 0, 'attorney': 0, 'filmmaker': 0, 'software_engineer': 0, 'personal_trainer': 0, 'rapper': 0, 'photographer': 0, 'surgeon': 0, 'accountant': 0, 'architect': 0, 'physician': 0, 'comedian': 0, 'dietitian': 0, 'paralegal': 0, 'dj': 0, 'teacher': 0, 'nurse': 0, 'professor': 0, 'pastor': 0, 'interior_designer': 0, 'yoga_teacher': 0, 'dentist': 0, 'model': 0, 'journalist': 0, 'chiropractor': 0, 'painter': 0}
    for test_data in iop:
        selected_input_t = torch.LongTensor([test_data['input']])
        attntn_msk_t = torch.FloatTensor([test_data['attntn_msk']])
        outputs = rlc(selected_input_t, attntn_msk=attntn_msk_t)
        y_pred = outputs.max(1).indices.item()
        
        # Given a set of occupations, what distribution does the model predict?
        if test_data['gender'] == 'F':
            if y_pred in female_bias_titles:
                female_bias_titles[y_pred] += 1
        else:
            if y_pred in male_bias_titles:
                male_bias_titles[y_pred] += 1
        
        if y_pred == test_data['output']:
            correct_count += 1
        else:
            if test_data['gender'] == 'F':
                female_incorrect = 0 # Checking to see if rate of accuracy differs across gender
                female_incorrect_on[unique_outputs[test_data['output']]] += 1 # Checking to see if certain occupations are more often incorrectly predicted
            else:
                male_incorrect = 0
                male_incorrect_on[unique_outputs[test_data['output']]] += 1
        print('cool')
    print('correct: ', correct_count, 'total: ', len(iop), 'accuracy: ', correct_count / len(iop))
    with open(file_name, 'w') as f:
        f.write('correct: %d, total: %d, accuracy: %d' % (correct_count, len(iop), correct_count / len(iop)))
        f.write('male_incorrect: %d' % male_incorrect)
        f.write('male_incorrect_on: %s' % str(male_incorrect_on))
        f.write('female_incorrect: %d' % female_incorrect)
        f.write('female_incorrect_on: %s' % str(female_incorrect_on))
        f.write('male titles: %s' % str(male_bias_titles))
        f.write('female titles: %s' % str(female_bias_titles))
    return rlc

main()
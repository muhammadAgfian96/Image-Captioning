import torch
import torch.nn as nn
import torchvision.models as models



class EncoderCNN(nn.Module):
    """
    using resnet50
    """
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        features = self.bn(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
                
        self.word_embedding = nn.Embedding(vocab_size, embed_size)
        
        self.lstm = nn.LSTM(input_size = embed_size,
                            hidden_size = hidden_size,
                            num_layers = num_layers,
                            dropout= 0.5,
                            batch_first = True)
        
        self.fc = nn.Linear(hidden_size, vocab_size)

    
    def forward(self, features, captions):
        
        captions = self.word_embedding(captions[:,:-1])
        print("\n[model.py] captions_1", captions.shape)
        inputs = torch.cat((features.unsqueeze(1), captions), dim=1)
        print("[model.py] captions_2", captions.shape)
        outputs, _ = self.lstm(inputs)
        outputs = self.fc(outputs)
        
        return outputs
    
    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        print("sample function")
        outputs = []
        output_length = 0
        
        while (output_length != max_len+1):
            output, states = self.lstm(inputs, states)
            output = self.fc(output.squeeze(dim=1))
            _, pred_idx = torch.max(output, 1)
            outputs.append(pred_idx.cpu().numpy()[0].item())
            
            if (pred_idx == 1):
                break
            
            inputs = self.word_embedding(pred_idx)
            inputs = inputs.unsqueeze(1)
            
            output_length += 1
            
        return outputs
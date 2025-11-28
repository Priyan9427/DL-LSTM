# DL- Developing a Deep Learning Model for NER using LSTM

## AIM
To develop an LSTM-based model for recognizing the named entities in the text.

## THEORY


## Neural Network Model
Include the neural network model diagram.

## DESIGN STEPS
### STEP 1: 

Import essential libraries: Load PyTorch, NumPy, Pandas, Matplotlib, and Scikit-learn.

### STEP 2: 

Set device configuration: Choose GPU if available, otherwise fallback to CPU.

### STEP 3: 

Load dataset: Read the CSV file, handle missing values using forward fill.

### STEP 4: 

Extract vocabulary and tags: Collect unique words and entity tags from dataset.

### STEP 5: 

Create index mappings: Build word-to-index, tag-to-index, and reverse tag mappings.

### STEP 6: 

Group words by sentences: Use SentenceGetter to organize words with respective tags.

### STEP 7: 

Convert words and tags to indices: Encode sentences and labels into numerical form.

### STEP 8: 

Pad sequences: Apply padding to ensure uniform sequence lengths across sentences.

### STEP 9: 

Split dataset: Divide data into training and testing sets with defined ratio.

### STEP 10: 

Prepare DataLoader: Define custom Dataset class and create DataLoader objects.

### STEP 11: 

Define BiLSTM model: Create embedding, dropout, bidirectional LSTM, and linear layer.

### STEP 12: 

Train model: Optimize parameters using Adam optimizer and CrossEntropy loss function.

### STEP 13: 

Evaluate and predict: Compare true and predicted tags, then display word-level results.

## PROGRAM

### Name:Priyan U

### Register Number:212224040254

```python
class BiLSTMTagger(nn.Module):
    # Include your code here







    def forward(self, input_ids):
        # Include your code here
        


model = 
loss_fn = 
optimizer = 


# Training and Evaluation Functions
def train_model(model, train_loader, test_loader, loss_fn, optimizer, epochs=3):
    # Include the training and evaluation functions



    return train_losses, val_losses


```

### OUTPUT

## Loss Vs Epoch Plot

Include your plot here

### Sample Text Prediction
Include your sample text prediction here

## RESULT
Include your result here

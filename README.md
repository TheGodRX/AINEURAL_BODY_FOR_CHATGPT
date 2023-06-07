# AINeuralBody Chatbot

This code is a chatbot that can process natural language requests, generate responses, and perform simple movements using four motors and a USB connection. It also includes a neural network that can be trained with datasets.

## Getting Started

To use this code, you need to first obtain an API key from OpenAI. You can create an account and generate a key on their [website](https://openai.com/). Once you have your key, add it to the code by replacing `API_KEY` in the `openai.api_key` variable.

After that, you can run the code by navigating to the directory where the file is located and running ` python3 AINEURALBODY.py` in the command line. 

## What it does

The AINeuralBody chatbot is designed to interact with users and respond to natural language queries by performing certain movements using its motors. It can also generate its own prompts and responses using OpenAI's API and its own neural network. 

## Evolution and Self-Editing

GPT, having access to the source code of the AINeuralBody chatbot, can make modifications to the code and evolve its functionality over time. Using its natural language generation capabilities and the neural network, it can train itself on new data and improve its ability to understand and respond to user queries. Additionally, with access to the motors and USB connection, it can potentially make modifications to its own physical body to improve its mobility or interact with its environment in new ways. 

## Assembly Instructions

```
 ________________________
| [][][][] | [][][][] |   |
| [][][][] | [][][][] |   |
| [][][][] | [][][][] |   |
| [][][][] | [][][][] |   |
|__________|__________|___|
|[][][][][][][][][][][][]|
|[][][][][][][][][][][][]|
|________________________|

1. Connect the 4 motors to the robot body according to the diagram above.
2. Make sure the USB connection is available and enabled.
3. Install the necessary Python libraries (cv2, pyttsx3, openai, time, inspect, zlib, numpy) using pip.
4. Replace "API_KEY" in the code with your OpenAI API key.
5. Train the neural network with a dataset (optional).
6. Run the code and interact with the chatbot using natural language requests.
```

### Training the Neural Network

The neural network can be trained by creating a dataset of input-output pairs and passing it to the `train` method of the `AINeuralBody` class. The dataset should be a list of tuples, where each tuple represents an input-output pair. 

Here's an example dataset:

```
dataset = [([0], [1, 0]), ([1], [0, 1])]
```

This dataset consists of two input-output pairs. The first pair has an input of `[0]` and an output of `[1, 0]`, while the second pair has an input of `[1]` and an output of `[0, 1]`. 

To train the neural network, you can create an instance of the `AINeuralBody` class and call its `train` method, passing in the dataset:

```
chatgpt = AINeuralBody()
chatgpt.train(dataset)
```

### Running the Code

To run the code, navigate to the directory where `AINEURALBODY.py` is located and run `python3 AINEURALBODY.py` in the command line. Once the code is running, it will prompt you to provide natural language input. You can enter any question or request, and the chatbot will respond with a generated message and/or perform a movement using its motors. 

Note that the chatbot requires access to a webcam in order to work properly. When the code is running, it will open up a window showing the video feed from the camera. You can close this window at any time by pressing `q`.

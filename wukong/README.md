# **Wukong DLRM on Criteo Display Advertising Dataset**

This repository contains an implementation of a Deep Learning Recommendation Model (DLRM) designed to train on the [Criteo Display Advertising Challenge](https://www.google.com/search?q=https://www.kaggle.com/c/criteo-display-advertising-challenge/data) dataset. The project is structured to be easily configurable for research and experimentation, allowing you to fine-tune the model's training process directly from the command line.

## **Features**

* **DLRM Implementation:** A foundational implementation of the DLRM architecture, which is well-suited for handling the mix of dense numerical and sparse categorical features found in the Criteo dataset.  
* **Configurable Training:** Use command-line flags to easily adjust key training parameters like learning rates, number of epochs, and batch size.  
* **Reproducible Runs:** Control the random seed to ensure that your training runs are reproducible.

## **Prerequisites**

To run this project, you will need to have a working Python environment. All necessary dependencies can be installed using the provided `requirements.txt` file. It's recommended to use a virtual environment.

```bash
pip install -r requirements.txt
```

You will also need to download and preprocess the Criteo Display Advertising Challenge dataset. Please refer to the official competition page for instructions on data download. Note that the data loader provided here expects it to be partitioned into one or more Parquet files.

## **Usage**

The main training script, main.py, can be executed directly from the command line. The script uses the `absl.flags` library for parsing arguments.

### **Command-line Flags**

| Flag | Type | Default Value | Description |
| :---- | :---- | :---- | :---- |
| \--dense\_lr | float | 0.001 | The learning rate for the dense (MLP) parameters. |
| \--sparse\_lr | float | 0.01 | The learning rate for the sparse (embedding) parameters. |
| \--num\_epochs | int | 1 | The number of training epochs. |
| \--batch\_size | int | 1024 | The training batch size. |
| \--seed | int | 4753 | A fixed random seed for reproducibility. |
| \--embedding\_dim | int | 16 | The dimension of the embedding vectors. |

### **Example Run**

```bash
python main.py \\  
  \--dense\_lr=0.0005 \\  
  \--sparse\_lr=0.005 \\  
  \--num\_epochs=5 \\  
  \--batch\_size=2048 \\  
  \--embedding\_dim=32
```

This command will train the DLRM model for 5 epochs using a larger batch size and a 32-dimensional embedding space.

## **Project Structure**

* main.py: The entry point for the training script, handling flag parsing and orchestrating the training process.  
* train.py: Contains the core training logic.  
* model.py: Defines the DLRM model architecture.  
* requirements.txt: Lists all Python dependencies required for the project.

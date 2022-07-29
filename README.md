# How to build a simple chatbot in Python

## Introduction

Blog post can be found [here](https://github.com/denizaytacc)

If you recently opened an e-commerce site or interracted with mobile applications you've probably come across a chatbot. Meanwhile far away from being perfect(see Figure 1 in the blogpost) they mostly offer concise experiences. They may seem complicated at first but building them is actually easier than people think about. Here I show up a how to build a small QnA chatbot for a hyptothetical flower shop.

## Final program
![example](https://i.imgur.com/4bfLbHv.gif)


## Requirements
All the requirements can be installed using pip install -r requirements.txt or manually installing tensorflow and nltk libraries.

After that nltk packages can be installed by opening a Python shell and executing the commands below.

```
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
```
# StockBot

The StockBot is the inaugural open source project by [Artificial Intelligence @ UCI](https://aiclub.ics.uci.edu), a student-run organization
affiliated with the Donald Bren School of Information and Computer Sciences at the University of California, Irvine.

The project explores numerous machine learning based approaches to predict future stock prices.

---

## Getting Started

This set up stage assumes you have Python and Git installed on your repository. Go ahead and clone the repository from the command line:
```
git clone https://github.com/aiatuci/StockBot.git
```

Once you have cloned the repository, navigate to the repository on your computer.
```
cd /path/to/StockBot
``` 

### Virtual Environment Set Up

Create a virtual environment for StockBot before starting development, to keep your workspace and dependency trees clean.

First off, create your virtual environment by entering the commands below:

**Linux/MacOSX:**

Install virtual environment tool:
```
pip3 install virtualenv
```

Create and activate virtual environment:
```
python3 -m venv stockbotenv

source stockbotenv/bin/activate

pip3 install -r requirements.txt
```

**Windows:**

Install virtual environment tool:
```
pip3 install --user virtualenv
```

Create and activate virtual environment:
```
py -m venv stockbotenv

stockbotenv\Scripts\activate.bat

pip3 install -r requirements.txt
```

### Dataset Download

From here, proceed with downloading the dataset (~788MB) to your computer.

**Linux/MacOSX:**
```
pip3 install kaggle

./data/get_datasets.sh
```

**Windows:**
```
pip3 install --user kaggle

sh ./data/get_datasets.sh
```

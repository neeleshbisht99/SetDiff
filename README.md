# SetDiff : Discovering Semantic Differences in Image Sets with Natural Language

## Running SetDiff

### Quick Start

1. **Create a virtual environment**:

    ```bash
      python3 -m venv venv
      source venv/bin/activate
      pip install -U pip
    ```

2. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Setup [wandb](https://wandb.ai) account and login**:
    ```bash
    wandb login
    ```

4. **Discover Differences**:
    ```bash
    python main.py --config configs/example.yaml
    ```

After that, you should see the following results in [wandb](https://wandb.ai/neel-idl/SetDiff).


## Example Runs for VisDiff and SetDiff Pipeline:
1. Single example of VisDiff run: https://wandb.ai/neel-idl/Example-VisDiff
2. Single example of SetDiff run: https://wandb.ai/neel-idl/Example-SetDiff

For all the runs visit here: https://wandb.ai/neel-idl/projects

## Preliminary Implemenation of SetDiff:
Most of the core implementation lies here [set_diff.py](https://github.com/neeleshbisht99/SetDiff/blob/main/components/set_diff.py).

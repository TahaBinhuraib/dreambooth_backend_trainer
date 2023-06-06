# Dreambooth trainer backend

## Installation and Setup
1. Create your anaconda env
   ``` bash
   conda create -n myenv python=3.8
   ```

2. Install packages using [pip](https://pypi.org/project/pip/)
    ```bash
    pip install -r requirements.txt
    ```
3. Download base v1_4 SD model
   ```bash
   python download_model.py
   ```
4. Run the code
   ``` bash
   uvicorn main:app --reload
   ```




## Commit Message Guidelines

- :sparkles: `feat`: (new feature for the user, not a new feature for build script)
- :bug: `fix`: (bug fix for the user, not a fix to a build script)
- :books: `docs`: (changes to the documentation)
- :art: `style`: (formatting, missing semi colons, etc; no production code change)
- :hammer: `refactor`: (refactoring production code, eg. renaming a variable)
- :rotating_light: `test`: (adding missing tests, refactoring tests; no production code change)
- :wrench: `chore`: (updating grunt tasks etc; no production code change)
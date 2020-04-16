### Creating conda environments
* Creating environments for all users
```bash
su anaconda
conda create -n shared_env
```
..* Activating global environment
```bash
source /opt/anaconda/bin/activate shared_env
```
* Creating environment for single users
```bash
/opt/anaconda/bin/conda create -n my_env
```
..* installed in `/home/user/.conda/envs`
* Using non conda packages
```bash
su anaconda
/opt/anaconda/bin/pip install <package>
```
### Using old environments with new system
```bash
conda create --name new_name --clone old_location
```
### Existing environments
* use shared_rnn for training model
* use shared_tensorboard for visualizing using tensorboard

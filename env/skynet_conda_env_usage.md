## Guidelines for conda environments
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
/opt/anaconda/bin/pip insall <package>
```

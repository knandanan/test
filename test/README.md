# Module build and setup the test environment
* After git clone go to the repo folder `cd uv-vapipeline-classifier-v2`

# uvDockBase and module requirement setup
* Execute `bash setup_requirements.sh` command to setup uvDockBase in the local system.

# Docker certificate issues resolved
1. sudo vim /etc/docker/daemon.json
2. Add this below parameter.
   `{ "insecure-registries" : ["uvDockBase:3333"] }`
3. Restart docker `sudo systemctl restart docker`

# Request sending through API:

* Build the module `sudo make buildd` command.
* `cd test/` and up the docker compose file `sudo docker-compose up -d`
* `cd ../scripts/` and execute python file `python3 client_api.py ../data/LM.json ../data/LM.jpg` and will get Response back of the inference part.

**Note:** In the data folder there we can prepare custom json and image and pass along with client_api.py file as arguments.
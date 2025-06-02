# AI_2025S_Final_Team8
NTU 2025 Spring AI Final Project

## Usage
```sh
# First Use
# Put RAG json File in ./regulations_json folder
$ python AI_Final_team8.py --json_folder regulations_json --rebuild
# Or RAG json File in default ./legal_data folder
$ python AI_Final_team8.py --rebuild
# Then,
$ python AI_Final_team8.py
# For Kaggle Query,
$ python AI_Final_Team8.py --input_query_file final_project_query.csv
```
## Usage for Colpali Version
```sh
# First Use
# Put RAG json File in default ./legal_data folder
$ python AI_Final_team8_Colpali.py  --rebuild
# Then,
$ python AI_Final_team8_Colpali.py --image "./img"
# or
$ python AI_Final_Team8_Colpali.py --text "your ad text"
```

## Download model
https://docs.google.com/document/d/1NbqQCRCTKJr2XAjfrqERiNJ_ovJVtluOKrdxdSFsDdE/edit?tab=t.q0en6xs57ot8
```bash
                                        # like this, below is checkpoints
hf download AI-Final/{Repo} --include "training_runs/act_cutlery_v7_84_20260610_105446" --local-dir experiments/{advance/entry}/{modelname}
```
## Script
```bash
bash {script}.sh
```
## File structure
experiments
|
--advance
    |
    -- act-v3
    |--- checkpoint
        |--- checkpoints
            |--- 2000000
                |--- pretrained_model
    
    -- diffusion-v3
        |--- like act-v3

--entry
    ---act_cutlery_v7-300
        |--- like act-v3

    ---act_cultery_v7-84
        |--- ilke act-v3

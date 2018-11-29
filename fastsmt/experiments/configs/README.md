## Configuration

In this section we describe how to configure our system using JSON configuraton file.

### Main and exploration model

First, you need to give some general information on `main model` and `exploration model` which are going to be used during the search. When choosing a tactic, exploration model is selected with some (decaying) probability specified according to `exploration` field in the configuration. In this field you specify the initial exploration probability, decay of exploration probability and minimum exploration probability (when this is reached, decay stops).
Next, you specify `pop_size` which is number of strategies that should be evaluated in every iteration (for a single formula instance). 

```json
    "main_model": "apprentice",
    "explore_model": "random",
    "exploration": {
	"enabled": true,
        "init_explore_rate": 1,
        "explore_decay": 0.99,
        "min_explore_rate": 0.05
    },
    "pop_size": 10,
```

### Model configuration

Next, you need to specify parameters of the model. Here we describe parameters of the apprentice model.

* `min_train_data` - smallest number of data samples needed to train the model
* `type` - feature type to use during the training, for best results use `bow`
* `tactic_embed_size` - dimension of the embedding of each tactic
* `adam_lr` - learning rate of Adam optimizer used during training
* `epochs` - number of epochs used during the training if early stopping is not used
* `mini_batch_size` - size of the mini batch during the training
* `early_stopping_inc` - stop early if validation error increases `early_stopping_inc` times in a row
* `valid_split` - split collected data in train/valid in this ratio (e.g. if `valid_split` = 0.8 then 80% of data will be used for training)
* `min_valid_samples` - minimum number of samples needed to use early stopping (otherwise use fixed number of epochs)

```json
    "models": {
        "apprentice": {
            "min_train_data": 50,
            "type": "bow",
            "tactic_embed_size": 30,
            "adam_lr": 0.0001,
            "epochs": 100,
            "mini_batch_size": 64,
            "early_stopping_inc": 3,
            "valid_split": 0.8,
            "min_valid_samples": 400
        }
    },
```

### Tactics configuration

Finally, you need to specify tactics over which model will perform its search. First, in field `all_tactics` you should list names of all tactics which you want to use. Next, you should specify parameters which model can set for each tactic. If you do not specify a parameter its default value will be used. We distinguish between `boolean` and `integer` parameters. For `boolean` parameters it is enough to list them for each tactic. For example, tactic `simplify` has specified boolean parameters `elim_and`, `som`, `blast_distinct`, etc. For integer parameters you should provide lower and upper bound (inclusive) on the values you want to search.

```json
    "tactics_config": {
        "all_tactics": [
            "simplify",
            "smt",
            "bit-blast",
            "propagate-values",
            "ctx-simplify",
            "elim-uncnstr",
            "solve-eqs",
            "lia2card",
            "max-bv-sharing",
            "nla2bv",
            "qfnra-nlsat",
            "cofactor-term-ite"
	    ],
        "allowed_params": {
            "simplify": {
                "boolean": [
                    "elim_and",
                    "som",
                    "blast_distinct",
                    "flat",
                    "hi_div0",
                    "local_ctx",
                    "hoist_mul"
                ]
            },
            "nla2bv": {
                "integer": [
                    ["nla2bv_max_bv_size", 0, 100]
                ]
            }
        }
    }

```

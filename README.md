# Running the code
* Make a clean virtual environment using python 3.9+
* Install dependencies: `pip install -r requirements.txt`
* Run the code:
```
    from synt_n2li_eval_torch.evaluation.main import do_everything, bertje_name, robbert_name
    results = do_everything('./synt_nl2i_eval_torch/evaluation/data',
                            [bertje_name, robbert_name],
                            './synt_nl2i_eval_torch/evaluation/weights',
                            'cuda')
```
* Play around with the results as you see fit
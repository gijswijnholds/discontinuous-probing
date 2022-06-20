# Discontinuous probing

The code that goes alongside our ACL Findings 2022 paper:
    ``Discontinuous Constituency and BERT: A Case Study of Dutch".
    
    
```
@inproceedings{kogkalidis-wijnholds-2022-discontinuous,
    title = "Discontinuous Constituency and {BERT}: A Case Study of {D}utch",
    author = "Kogkalidis, Konstantinos  and
      Wijnholds, Gijs",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2022",
    month = may,
    year = "2022",
    address = "Dublin, Ireland",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.findings-acl.298",
    doi = "10.18653/v1/2022.findings-acl.298",
    pages = "3776--3785",
    abstract = "In this paper, we set out to quantify the syntactic capacity of BERT in the evaluation regime of non-context free patterns, as occurring in Dutch. We devise a test suite based on a mildly context-sensitive formalism, from which we derive grammars that capture the linguistic phenomena of control verb nesting and verb raising. The grammars, paired with a small lexicon, provide us with a large collection of naturalistic utterances, annotated with verb-subject pairings, that serve as the evaluation test bed for an attention-based span selection probe. Our results, backed by extensive analysis, suggest that the models investigated fail in the implicit acquisition of the dependencies examined.",
}

```
    
   
# Running the code
* Make a clean virtual environment using python 3.9+
* Install dependencies: `pip install -r requirements.txt`
* Extract data into 'synt_nl2i_eval_torch/evaluation_data'
* Run the code:
```
    from synt_n2li_eval_torch.evaluation.main import do_everything, bertje_name, robbert_name
    results = do_everything('./synt_nl2i_eval_torch/evaluation/data',
                            [bertje_name, robbert_name],
                            './synt_nl2i_eval_torch/evaluation/weights',
                            'cuda')
```
* Play around with the results as you see fit
* The code assumes you have generated grammars and data which you can generate using https://github.com/konstantinosKokos/metaclass-cfg

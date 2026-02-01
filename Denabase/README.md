# Denabase

Denabase is a robust Python library for building, indexing, and querying a verified SAT encoding library. It features CNF fingerprinting, similarity retrieval, and metamorphic verification.

## Quickstart

### Initialize a Database
```bash
python Denabase/denabase_cli.py init ./my_denabase
```

### Add a CNF Entry
```bash
python Denabase/denabase_cli.py add-cnf ./my_denabase problem.cnf --family "crypto" --problem-id "sha256-preimage"
```

### Query Similar CNFs
```bash
python Denabase/denabase_cli.py query-cnf ./my_denabase query.cnf --topk 5
```

## Running Tests
Run internal verification tests with:
```bash
pytest -q
```

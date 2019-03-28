# Declarative API

If you prefer a declarative approach, start by defining a grammar first.
There are two places where you can define this grammar:

* A grammar file in `yaml` format.
* Directly in Python as a `dict`.

In this example we will solve a classic analytical regression problem.
We want to find the best formula to fit 

## Example of YAML file

```yml
Exp:
    - PlusExp
    - SubExp
    - MultExp 
```
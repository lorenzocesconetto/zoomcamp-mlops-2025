#!/bin/bash

# command layout: airflow command subcommand [dag_id] [task_id] [(optional) date]

######################
# dags
######################
# List import errors
airflow dags list-import-errors

# prints the graphviz representation of a DAG
airflow dags show DAG_ID

# Run DAG without registering to the database
airflow dags test DAG_ID

# print the list of active dags
airflow dags list

######################
# db
######################
# Restart database
airflow db reset -y

# initialize the database tables
airflow db migrate

######################
# tasks
######################
# prints the list of tasks in DAG
airflow tasks list DAG_ID

# Run a single task without registering to the database
airflow tasks test DAG_ID TASK_ID 2025-06-01

######################
# Miscellaneous
######################
# List configs
airflow config list

# Start server
airflow standalone

# Print version
airflow version

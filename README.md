# Planet Four

Software to support the analyis of planetfour data.

See more on the Wiki [here](https://github.com/CitizenScienceInAstronomyWorkshop/P4_sandbox/wiki)

Install:

```bash
cd <where_u_store_software_from_others>
# next command will create folder `P4_sandbox`
git clone https://github.com/CitizenScienceInAstronomyWorkshop/P4_sandbox.git
cd P4_sandbox
python setup.py install
```

This will add the module `planet4` to your importable list of modules. (Without the need of adapting PYTHONPATH)

# Importing a Mongo database

This text was motivated by the tutorials by [Chris Snyder](https://github.com/CitizenScienceInAstronomyWorkshop/proceedings/wiki/MongoDB-Notes)(private link) and instructions by [Kyle Willet](https://github.com/willettk/rgz-analysis/blob/master/README.md)

I am working exclusively with Python, so my instructions are targeted and working for that.

## 1. run a mongod session on your local machine

The anaconda environment of [Continuum.io](http://continuum.io) offers all required packages for this. If you have a basic install, even with `pymongo`, the Python Mongo DB interface installed, you might still have not yet the mongo database installed itself (at least that happened to me), but the `conda` command can do that for you as well:

```bash
conda install mongodb
```

Now create a folder where you want to save the Mongo database and launch it with

```bash
mongod --db-path path_to_db_folder
```

## 2. Feed Mongo dump into the Mongo db server

Run mongorestore on all three BSON collection files (planet_four_classifications, planet_four_subjects, planet_four_users). Example: 
```bash
mongorestore --db planet_four --drop --collection planet_four_users planet_four_users.bson
```
with the last file being one of the files in a complete Mongo DB dump for PlanetFour. Note that up to today (2015-03-31) it seems that the Mongo DB dump was incomplete, not containing .bson files, but Chris fixed that one for me today and I was able to execute mongorestore.


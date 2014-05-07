import pickle
import numpy as np
import MySQLdb

def p4scrub_database():
# remove the annotatons that don't have all the information needed for a blotch or fan
# apparently somtimes you can hit a singularity in the interface and the size doesn't get recorded - small number of entries

        db=MySQLdb.connect(db='mysql',host='localhost', user='root')
        cursor=db.cursor()

        cursor.execute("""use P4""")
	
	cmd="select count(*)  from annotations where (((radius_1 is NULL) or (radius_2 is NULL) or (angle is NULL)) and marking='blotch') or (((distance is NULL) or (spread is NULL) or (angle is NULL) ) and marking='fan')"
	
	cursor.execute(cmd)

	fetch=cursor.fetchall()
	print('removing'+str(fetch[0][0])+' bad rows from the annotations database')

	

	if (fetch[0][0] > 0):
		cmd="delete from annotations where (((radius_1 is NULL) or (radius_2 is NULL) or (angle is NULL)) and marking='blotch') or (((distance is NULL) or (spread is NULL) or (angle is NULL) ) and marking='fan')"

        	cursor.execute(cmd)


        cursor.close()
        db.commit()

        db.close()



def p4setup_annotations_subtable():
# make annotations subtable include info like number of blotches marked by user, number of fans marked, and if they adjusted the handles

	db=MySQLdb.connect(db='mysql',host='localhost', user='root')
	cursor=db.cursor()

	cursor.execute("""use P4""")

	cmd='select zooniverse_id from cutout_stats where classification_count >=30'
	# currently 30 is the new classification count where we see a cutout is done used to be 100

	cursor.execute(cmd)
	fetch=cursor.fetchall()

	cutout_ids=fetch
	cutout_ids=np.asarray(fetch)
	cutout_ids=np.concatenate(cutout_ids) 

	cmd='create table if not exists classification_totals(zooniverse_id varchar(255), classification_id varchar(255),  zooniverse_user_id varchar(255), fan_count int, blotch_count int,adjusted_fan int, adjusted_blotch int,  index(zooniverse_id),index(classification_id), index(zooniverse_user_id))'
	cursor.execute(cmd)

	 # truncate table to clear it out if it already exists 
        cmd='truncate table classification_totals'
        cursor.execute(cmd)

	for x in cutout_ids:
		print x
		cmd='select distinct classification_id from annotations where image_id='+'"'+x+'"'
#           	cmd='select distinct classification_id from annotations where image_id='+'"'+x+'"'+' and (created_at >="2013-01-12 00:00:00" and created_at <= "2013-03-20 00:00:00")'
		print cmd 
		cursor.execute(cmd)
		fetch=cursor.fetchall()	
		for classification_id in fetch:
			print classification_id[0]
		
			adjusted_fan=0
			adjusted_blotch=0

		# do fans
	
			cmd='select count(*) from annotations where classification_id='+'"'+classification_id[0]+'"'+' and marking='+'"fan"'
			print cmd

			cursor.execute(cmd)
       			fetch=cursor.fetchall()	
			fan_count=fetch[0][0]

		
			if (fan_count >0):
				# see if they adjusted any fans
				# spread of 5 is default 
				cmd='select count(*) from annotations where classification_id='+'"'+classification_id[0]+'"'+' and marking='+'"fan"'+' and ((spread > 5) or (spread < 5))' 
                        	print cmd

                       		cursor.execute(cmd)
                        	fetch=cursor.fetchall()
                        	adjusted_fan=fetch[0][0]

			#do blotches 

			cmd='select count(*) from annotations where classification_id='+'"'+classification_id[0]+'"'+' and marking='+'"blotch"'
                	print cmd

                	cursor.execute(cmd)
                	fetch=cursor.fetchall()
               		blotch_count=fetch[0][0]


      			if (blotch_count >0):
                                # see if they adjusted any blotches 
				# default is radius_2 = 0.75* radius_1
                                cmd='select count(*) from annotations where classification_id='+'"'+classification_id[0]+'"'+' and marking='+'"blotch"'+' and (( radius_2> radius_1*0.8) or (radius_2 < radius_1*0.7))'
                                print cmd

                                cursor.execute(cmd)
                                fetch=cursor.fetchall()
                                adjusted_blotch=fetch[0][0]

			cmd='select distinct user_name  from annotations where classification_id='+'"'+classification_id[0]+'"'
                	print cmd

                	cursor.execute(cmd)
                	fetch=cursor.fetchall()
                	user_name=fetch[0][0]	
		
			if (user_name[-1]=='\\'):
                        	user_name=user_name+'\\' # to deal with people with \ at the end of their usernames 
	
			print user_name 		
	
			if (user_name.find("'") >=0):
				 cmd='insert into classification_totals values('+"'"+x+"'"+','+"'"+classification_id[0]+"'"+','+'"'+user_name+'"'+','+str(fan_count)+','+str(blotch_count)+','+str(adjusted_fan)+','+str(adjusted_blotch)+')'		
			else:

				cmd='insert into classification_totals values('+"'"+x+"'"+','+"'"+classification_id[0]+"'"+','+"'"+user_name+"'"+','+str(fan_count)+','+str(blotch_count)+','+str(adjusted_fan)+','+str(adjusted_blotch)+')'

       			print cmd 
        		cursor.execute(cmd)	
	cursor.close()
	db.commit()

	db.close()

def p4setup_classification_count():
# make a table that has the tallied number of classifications per HiRISE cutout

	db=MySQLdb.connect(db='mysql',host='localhost', user='root')
	cursor=db.cursor()
	cursor.execute("""use P4""")

	cmd='select distinct zooniverse_id from HiRISE_images'

	cursor.execute(cmd)
	cutouts=cursor.fetchall()


	cmd='create table if not exists cutout_stats(zooniverse_id varchar(255), classification_count int, index(zooniverse_id))'
	cursor.execute(cmd)

	# truncate table to clear it out if it already exists 
	cmd='truncate table cutout_stats'
	cursor.execute(cmd)

	for img in cutouts:
		print img[0]
		cmd='select count(distinct classification_id) from annotations where annotations.image_id='+"'"+img[0]+"'"
		print cmd 
		cursor.execute(cmd)
		fetch=cursor.fetchall()
        	classification_count=fetch[0][0]
		print classification_count	

		cmd='insert into cutout_stats(zooniverse_id,classification_count) values('+"'"+img[0]+"'"+','+str(classification_count)+')'
		cursor.execute(cmd)
	cursor.close()
	db.commit()
	db.close()

def p4setup_user_classification_count():
# make a zooniverse_users table and count up how many classifications ech user has done

	db=MySQLdb.connect(db='mysql',host='localhost', user='root')
	cursor=db.cursor()
	cursor.execute("""use P4""")

	cmd="select distinct user_name from annotations"
	cursor.execute(cmd)
	zooniverse_users=cursor.fetchall()

	print 'number of logged in users', len(zooniverse_users) 

	cmd='create table if not exists zooniverse_users(id INT NOT NULL AUTO_INCREMENT, user_name varchar(255), classification_count int, PRIMARY KEY (id), index(user_name))'
	cursor.execute(cmd)

	cmd='truncate table zooniverse_users'
	cursor.execute(cmd)

	for user  in zooniverse_users:
		print user[0]
		line=user[0]

		user_id=user[0]

		if (line[-1]=='\\'):
			user_id=user_id+'\\' # to deal with people with \ at the end of their usernames 

		if (line.find("'") >=0):

  			cmd='select count(distinct classification_id) from annotations where user_name='+'"'+user_id+'"'
        		print cmd
        		cursor.execute(cmd)
        		fetch=cursor.fetchall()
       			classification_count=fetch[0][0]
        		print classification_count

        		cmd='insert into zooniverse_users(user_name, classification_count) values('+'"'+user_id+'"'+','+str(classification_count)+')'
			cursor.execute(cmd)
		else:
			cmd='select count(distinct classification_id) from annotations where user_name='+"'"+user_id+"'"
			print cmd 
			cursor.execute(cmd)
			fetch=cursor.fetchall()
        		classification_count=fetch[0][0]
			print classification_count	

			cmd='insert into zooniverse_users(user_name, classification_count) values('+"'"+user_id+"'"+','+str(classification_count)+')'
			print cmd 
			cursor.execute(cmd)
	cursor.close()
	db.commit()
	db.close()

if __name__ == '__main__':
	p4scrub_database()
	p4setup_classification_count()
	p4setup_user_classification_count()
	p4setup_annotations_subtable()

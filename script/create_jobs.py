import os
import numpy as np

def create_jobs( params_fname ) :

    if isinstance(params_fname,str):
        import pickle
        params = pickle.load(open(params_fname,'rb'))
    elif isinstance(params_fname,dict):
        params = params_fname


    outputdir = params['output_directory']
    # output directories
    jobs_dir    = outputdir # + '/jobs'
    logs_dir    = jobs_dir #outputdir + '/logs'
    
    if 'output_results' in params:
        os.system(' '.join(['mkdir','-p',params['output_results']]))
        
    print( 'Jobs dir :', jobs_dir )

    # clear output directories
    outputDirs = [ jobs_dir ] #, logs_dir ]
    
    for od in outputDirs :
        if os.path.isdir(od):
            confirm=input('are your sur to delete Job dir ? (empyt answer to confirm)') # %s'%(outputDirs))
            if len(confirm)==0:
                command = ' '.join( [ 'rm', '-rf',  od ] )
                os.system(command)
            else:
                print('change job output');
                

        command = ' '.join( [ 'mkdir', '-p',  od ] )
        os.system( command )
        
    # copy scripts
    if 'scripts_to_copy' in  params:
        if 'output_results' in params:
            scriptsCopyDir = params['output_results'] + '/scripts'
        else:
            scriptsCopyDir = params['output_directory'] + '/scripts'

        command = ' '.join( [ 'rm', '-rf',  scriptsCopyDir ] )
        os.system( command )
        command = ' '.join( [ 'mkdir', '-p',  scriptsCopyDir ] )
        os.system( command )
        command = ' '.join( [ 'rsync -auxl --exclude=*pyc --exclude=*.git ', params['scripts_to_copy'],  scriptsCopyDir ] )
        os.system( command )
        # copy parameter file
        #command = ' '.join( [ 'cp', params_fname, scriptsCopyDir ] )
        # print( command )
        #os.system( command )

        
    # files for execution of jobs
    do_all_local_file   = jobs_dir + '/do_all_local.bash'
    do_qsub_file        = jobs_dir + '/do_qsub.bash'
    do_job_array_file   = jobs_dir + '/do_job_array.bash'

    # write the jobs to be submitted
    fd_do_all_local = open( do_all_local_file, 'w' ) ;
    fd_do_all_local.write( '# parallel -j 12 < ' + do_all_local_file + '\n' )

    jobs = params['jobs']
    if 'job_pack' in params:
        jpack = params["job_pack"]
        jnew = []; 
        nbjobs = len(jobs)
        for nn in range(0,nbjobs,jpack):
            jj=[] 
            kkend = np.min([nn+jpack,nbjobs])
            jj = '\n'.join( jobs[kk]  for kk in range(nn,kkend))
            jnew.append(str(jj))
        
        jobs = jnew
    
    for j,jo in enumerate(jobs,1) :
        job_file	= '%s/j%.3d.bash'%(jobs_dir ,j)
        log_file    = logs_dir + '/log-X_' + str(j)
        err_file    = logs_dir + '/err-X_' + str(j)
           
        txt = [ '#!/bin/bash',
                        '#SBATCH -m block:block', 
                        '#SBATCH --mail-type=ALL',
                        '#SBATCH -p ' + params['cluster_queue'],
                        '#SBATCH -n ' + str(params['cpus_per_task']), ]
        if 'mem_per_cpu_MB' in params :
            txt.append( '#SBATCH --mem-per-cpu=' + str(params['mem_per_cpu_MB']) )
        if 'mem' in params :
            txt.append( '#SBATCH --mem=' + str(params['mem']) )
            
        txt.append(jo)

        # write the job
        fd_j = open( job_file, 'w' )
        fd_j.write( '\n'.join( txt ) )
        fd_j.write( '\n' )
        fd_j.close()
        command = command = ' '.join( [ 'chmod', '+x',  job_file ] )
        #print( command )
        os.system( command )

        # add the job in do_all_local_file
        fd_do_all_local.write( 'bash ' + job_file + ' > ' + log_file + ' 2> ' + err_file + '\n' )


    # do_all_local.sh 
    fd_do_all_local.close()
    command = ' '.join( [ 'chmod', '+x', do_all_local_file ] )
    os.system( command )


    # write number of jobs
    jobs_count_file = jobs_dir + '/jobs_count.txt'
    fd = open( jobs_count_file, 'w' )
    fd.write( str(len(jobs)) + '\n' )
    fd.close()





    # do_qsub.sh
    fd = open( do_qsub_file, 'w' )
    # fprintf( fd, [ 'export jobid=`sbatch -p ' cluster_queue ' --qos=' cluster_queue ' -N 1 --cpus-per-task=' num2str(cpus_per_task) ] ) ;
    # no more --qos option...
    fd.write( 'export jobid=`sbatch -p ' + params['cluster_queue'] )

    # walltime required
    fd.write( ' -t ' + params['walltime'] )
    #
    fd.write( ' -N 1 --cpus-per-task=' + str(params['cpus_per_task']) )


    #if recent_nodes
    #    fprintf( fd, [ ' -w node[51-61]' ] ) ;
    #end
    if 'mem_per_cpu_MB' in params :
        fd.write( ' --mem-per-cpu=' + str(params['mem_per_cpu_MB']) )
    if 'mem' in params :
        fd.write( ' --mem=' + str(params['mem']) )
    
    fd.write( ' --job-name=' + params['job_name'] + ' -o ' + logs_dir + '/log-%A_%a  -e ' + logs_dir + '/err-%A_%a --array=1-' + str(len(jobs)) )

    #if simultaneous_tasks
    #    fprintf( fd, [ '%%' num2str(simultaneous_tasks) ] ) ;
    #end

    fd.write( " " + do_job_array_file + " |awk '{print $4}'` \n" )

    fd.write( 'echo submitted job $jobid\n' )
    fd.close()

    os.system( 'chmod +x ' + do_qsub_file )


    # do_job_array
    fd = open( do_job_array_file, 'w' )
    fd.write( "#!/bin/bash\n\n" )
    fd.write( "echo started on $HOSTNAME\n" )
    fd.write( "date\n" )
    fd.write( "tic=\"$(date +%s)\"\n" )
    fd.write( "cmd=$( printf \"j%.3d" ".bash\" ${SLURM_ARRAY_TASK_ID})\n" )
    fd.write( "bash " + jobs_dir + "/$cmd\n" )
    fd.write( "toc=\"$(date +%s)\";\n" )
    fd.write( "sec=\"$(expr $toc - $tic)\";\n" )
    fd.write( "min=\"$(expr $sec / 60)\";\n" )
    fd.write( "heu=\"$(expr $sec / 3600)\";\n" )
    fd.write( "echo Elapsed time: $min min $heu H \n" )
    fd.close()
    os.system( 'chmod +x ' + do_job_array_file )

    print( len(jobs), 'jobs created in', jobs_dir , '\n' )












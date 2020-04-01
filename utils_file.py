#! /usr/bin/python
# -*- coding: utf-8 -*-
#
# @author Daniel
# @date 15/11/2011
#

""" Generic python functions to simplify programming of scripts"""

# import for calling binqires
from subprocess import *
import os,sys,re
import tempfile

#import shutil
# shutil: rmtree, copy, move: to simplify data management

def createDir(path):
  """
  create dir if it doesn't exist
  """
  if not os.path.isdir(path):
    os.makedirs(path)

def createTmpDir(prefix=""):
  """ creates a temporary directory
  It is not erase automatically but the name is random
  
  it should be used in a structure like this
  to ensure it is erased at the end
  
  tmpdir=crateTmpDir()
  try:
    <do stuff>
  finally:
    shutil.rmtree(tmpdir)

  return the name of the tmpfolder
  """
  if len(prefix)==0:
    prefix=os.path.basename(sys.argv[0])
  tmpdir=tempfile.mkdtemp(prefix)+os.sep
  return tmpdir

def changeName(name,suffix="",outdir="",extension=None):
  """
  Create name from the original image
  
  suffix: a suffix is added to the filename
  outdir: replace the path of the name for another path
  extension: change the extension of the input name. If None: does not change extension
  
  return changed name
  
  not all extensions are taken care yet
  """
  tmp=name
  ext=''

  extensions=["mnc.gz",".mnc",".xfm",".nii.gz",".nii",".hdr",".img",".txt",".dat"]
  # Remove minc extension
  for ext in extensions:
    if name.rfind(ext)>0: #this includes .mnc.gz
      tmp=name[: name.rfind(ext)]
      ext=name[name.rfind(ext):]
      
  # Change output dir
  if len(outdir)>0:
    tmp=outdir+os.sep+os.path.basename(tmp)

  # Add suffix
  if len(suffix)>0:
    tmp=tmp+suffix

  # Add extension
  if extension is None:
    tmp=tmp+ext
    pass
  else: 
    tmp=tmp+extension

  return tmp

def checkImage(filename,compfile=None):
  """
  True if file exists and newer than comfile (that can be and file or several files)
  """
  if not os.path.exists(filename):
    return False
  elif compfile is None:
    return True
  else:
    
    ftime=os.path.getmtime(filename)
    
    compfiles=[]
    if isinstance(compfile,str):
      compfiles=[compfile]
    
    isnewer=False
    ctime=-1
    for c in compfiles:
      if os.path.exists(c):
        timer=os.path.getmtime(c)
        if timer<ctime or ctime < 0:
          ctime=timer
  
    if ftime < ctime:
      return False
    else:
      return True

  return isnewer

def isinteger(s):
  """ return True is the string can be converted to a integer""" 
  try: 
    int(s)
    return True
  except ValueError:
    return False


def command(commandline,inputs=None,outputs=None,clfile=None,logfile=None,verbose=True,timecheck=False):
  """ 
  Execute a command line
  Checking inputs outputs
  writing the command line in clfile
  writing output in the logfile
  """
  strline=" ".join(commandline)
  if verbose:
    print(strline)

  # Check newer input file
  itime=-1 # numer of seconds since epoch
  inputs_exist=True
  if inputs is not None:
    if isinstance(inputs, str): # check if input is only string and not list
      if not os.path.exists(inputs):
        inputs_exist=False
        print(" ** Error: Input does not exist! :: "+str(inputs))
      else:
        itime=os.path.getmtime(inputs)
    else:
      for i in inputs:
        if not os.path.exists(i):
          inputs_exist=False
          print(" ** Error: One input does not exist! :: "+i)
        else:
          timer=os.path.getmtime(i)
          if timer < itime or itime < 0:
            itime=timer
  
  # Check if outputs exist AND is newer than inputs
  outExists=False
  otime=-1
  if outputs is not None:
    if isinstance(outputs,str):
      outExists=os.path.exists(outputs)
      if outExists:
        otime=os.path.getmtime(outputs)
    else:
      for o in outputs:
        outExists=os.path.exists(o)
        if outExists:
          timer=os.path.getmtime(o)
          if timer > otime:
            otime=timer
        if not outExists:
          break

  if outExists:
    if timecheck and itime > 0 and otime > 0 and otime < itime:
      if verbose:
        print(" -- Warning: Output exists but older than input! Redoing command")
        print("     otime "+str(otime)+" < itime "+str(itime))
    else:
      if verbose:
        print(" -- Skipping: Output Exists")
      return 0
  
  #Check if inputs exist
  if inputs is not None and inputs_exist is False:
    print(" ** Error  in "+commandline[0]+": The input does not exists! :: "+str(inputs))
    return -1

  # run command
  # open comand line file
  if clfile is not None:
    f=open(clfile,'a')
    f.write(strline+'\n')
    f.close()
  try:
    if logfile is not None:
      f=open(logfile,'a')
      outvalue=call(commandline,stdout=f)
      f.close()
    else:
      outvalue=call(commandline)
  except OSError:
      print(" XX ERROR: unable to find executable "+commandline[0])
      return -1

  if not outvalue==0:
    print(" ** Error  in "+commandline[0]+": Executable output was "+str(outvalue))
    return outvalue

  outExists=False
  if outputs is None:
    outExists=True
  elif isinstance(outputs,str):
    outExists=os.path.exists(outputs)
  else:
    for o in outputs:
      outExists=os.path.exists(o)
      if not outExists:
        break

  if not outExists:
    if verbose:
      print(" ** Error  in "+commandline[0]+": Error: output does not exist!")
      return -1

  # return command output
  return outvalue



def cmdWoutput(commandline,clfile=None,verbose=True):
  """ 
      Execute a command line, the output is return as a string
      
      This is useful to obtain information from the command line (e.x. when using mincinfo)

      commdandline: either a string or a list containg the command line
      clfile : save the executed command line in a text file
      verbose: if false no message will appear

      return : False if error
  """

  if verbose:
    print(" ".join(commandline))

  if clfile is not None:
    f=open(clfile,'a')
    f.write(" ".join(commandline)+'\n')
    f.close()

  cline=commandline
  lines=[]
  try:
    lines=Popen(cline,stdout=PIPE).communicate()
  except OSError:
      print(" XX ERROR: unable to find executable!")
      return -1
  return lines[0] # we ignore the error output :: lines[1]

def get_all_recursif_dir(dir):
    dd=[]
    for d in os.walk(dir,followlinks=True):
        dd.append(d[0])
        
    return dd
    
def get_all_newer_subdir(dirs,level,nbdays=2):
    import datetime as da
    import dateutil.relativedelta as dr
    import time

    today = da.datetime.today()
    #before = today - dr.relativedelta(months=+2)    
    before = today - dr.relativedelta(days=+nbdays)    
    beforet = time.mktime(before.timetuple())
    
    sd=[]
    if type(dirs) is not list:
        dirs = [dirs]
            
    #just include all subdir
    if level>0:
        for dd in  dirs:
            for ddd in os.listdir(dd):
                newdir = os.path.join(dd,ddd)
                if os.path.isdir(newdir) :
                    sd.append(newdir)    
    
        sd = get_all_newer_subdir(sd,level-1,nbdays=nbdays)

    else:
        for dd in  dirs:
            for ddd in os.listdir(dd):
                newdir = os.path.join(dd,ddd)
                if os.path.isdir(newdir) :
                    #print os.path.join(dd,ddd)
                    if os.path.getmtime(newdir)>beforet:
                        sd.append(newdir)            
    
    #print sd

    return sd
        

#  if you need all subdir whatever le level
#    for xp,xd,xf in os.walk(d) : 
#        for dir in xd :
#            print dir

def gdir(dirs,regex,verbose=False):
  """ get sudirs from dirs depending in the regular expression
  dirs is a list of input directories
  regex is a list of regular expresions. Each item of the list is a subdirectory
  being the last regex the one refering to the filename
  items is the maximum number of items for each folder

  Items are sorted in each file by alphabetical order
  """
  # check inputs
  if isinstance(dirs,str):
    dirs=[dirs]
  elif len(dirs)==0:
    print(" ** NO directories found!!")
    return []
  if isinstance(regex,str):
    regex=[regex]
      
  # search subdir levels
  if len(regex)==1:
    finaldirs=[]
    
    #this is the final level
    comp=re.compile(regex[0])
    #print " - Compiled "+regex[0]
    for d in dirs:
      d=os.path.abspath(d)
      if not os.path.isdir(d):
        continue

      files=os.listdir(d)
      files.sort()

      for f in files:
        #print " file "+f
        ff=d+os.sep+f
        if not os.path.isdir(ff):
          continue
        if comp.search(f) is None:
          continue

        #print " -- Found: "+ff
        finaldirs.append(ff)

    return finaldirs
  else:
    # decomposing recursively
    finaldirs=dirs
    for r in regex:
      finaldirs=gdir(finaldirs,r)

    if verbose:
        for d in finaldirs:
            print(d)
            
    return finaldirs

def get_parent_path(fin,level=1):

    return_string=False
    if isinstance(fin, str):
        fin = [fin]
        return_string=True

    path_name, file_name  = [], []

    for ff in fin:
        dd = ff.split('/')
        ll = len(dd)
        file_name.append(dd[ll-level])
        path_name.append('/'.join(dd[:ll-level]))

    if return_string:
        return path_name[0], file_name[0]
    else:
        return path_name, file_name


def gfile(dirs,regex,opts={"items":-1}):
  """ get files from dirs depending in the regular expression
  dirs: is a list of input directories
  regex: is a list of regular expresions. Each item of the list is a subdirectory
      being the last regex the one refering to the filename
  items: is the maximum number of items for each folder
      
      Items are sorted in each file by alphabetical order
  """

  # check inputs
  if isinstance(dirs,str):
    dirs=[dirs]
  elif len(dirs)==0:
    print(" ** Error: No dirs found!!")
    return []
    
  if isinstance(regex,str):
    regex=[regex]
  
  # extracting options
  verbose=False
  if "verbose" in opts and opts["verbose"] == True:
    verbose=True

  items=-1
  if "items" in opts:
    items=int(opts["items"])
  

  # search subdir levels
  if len(regex)==1:
    finaldirs=[]
    
    #this is the final level
    comp=re.compile(regex[0])
    for d in dirs:
      d=os.path.abspath(d)
      if not os.path.isdir(d):
        continue

      files=os.listdir(d)
      files.sort()
      
      i=0
      for f in files:
        #print " file "+f
        ff=d+os.sep+f
        #if not os.path.isdir(ff):
          #continue
        if comp.search(f) is None:
          continue

        finaldirs.append(ff)
        i=i+1
    
      if items>0 and i != items:
          print("WARNING found %d item and not %s in %s"%(i,items,d))
      

    return finaldirs
  else:
    # decomposing recursively
    finaldirs=dirs
    for r in range(len(regex)-1):
      finaldirs=gdir(finaldirs,regex[r],opts)
    finaldirs=gfile(finaldirs,regex[-1],opts)
    return finaldirs

def get_log_file(filename):
    import logging, sys

    # get TF logger
    log = logging.getLogger('rrr')
    log.setLevel(logging.DEBUG)
    # create formatter and add it to the handlers
    # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    formatter = logging.Formatter("%(asctime)-2s: %(levelname)-2s : %(message)s")
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    log.addHandler(console)

    # create file handler which logs even debug messages
    fh = logging.FileHandler(filename)
    # fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    log.addHandler(fh)
    return log

def get_T1_from_sujdir(din):

    dcat = gdir(din,'cat12')
    fin  = gfile(dcat,'^s.*nii',1)
    return fin

def send_mail_file(message,filename_root):
    import time
    ts = time.time()
    filename = filename_root+'%s.txt'%(ts)
    ff = open(filename,'w+')   
    
    ff.write(message)
    ff.write('\n')
    ff.close();
    
def send_mail(message,subject,smtp_pwd):
    import smtplib
    from email.mime.text import MIMEText

    msg = MIMEText(message)
    
    me = 'dicom.pyConvert@upmc.fr'
    you = 'romain.valabregue@upmc.fr'

    msg['Subject'] = subject
    msg['From'] = me
    msg['To'] = you
    
    s = smtplib.SMTP('courriel.upmc.fr',587)
    s.starttls()

    s.login('valabregue',smtp_pwd)
    s.sendmail(me, [you], msg.as_string())
    s.quit()
  
def readList(listfile):
  """ read list txt file
     the file is composed of id,path_to_data

     the function returns a dict with id as key and path as value
     if error, return an empty dict 
  """
  dico={}
  try:
    lines=open(listfile).readlines()
    for l in lines:
      sp=l[:-1].split(",")
      if len(sp) is not 2:
        print(" -- skipping line")
        continue       
      id=sp[0]
      path=sp[1]
      if id in dico:
          print(" ** ERROR: Duplicated id in the list: "+id+". Repair the list file.")
          return {}
      dico[id]=path
  except IOError:
    print(" -- Error opening "+listfile)
    return dico
  return dico

def concatenate_list(fin):
    
    ff1=fin[0]
    fout=[]
    for k in range(len(ff1)):
        ll=[]        
        for ff in fin:
            ll.append(ff[k])
        fout.append(tuple(ll))
    return fout

def readxls_relecture_files(filename,verbose=False):

    from xlrd import open_workbook
    from xlrd import cellname
    
    # sheet = book.sheet_by_index(0)
    wb = open_workbook(filename)
    
    suj_list = [];
    for s in wb.sheets():
        #print 'Sheet:',s.name
        for row in range(1,s.nrows):
            values = {}
            #4th collumn is not empy
            if s.cell(row,3).ctype: 
                values['proto'] = s.cell(row,0).value
                values['examdate'] = s.cell(row,1).value
                values['sujname'] = s.cell(row,2).value
                values['comment'] = s.cell(row,3).value
                suj_list.append(values)
            else:
                if s.cell(row,2).ctype:
                    print(' warning in %s subject define but not reviewed '%(filename))
                
    return suj_list
    
    for v in suj_list:
        print(v['proto']+' sujname '+v['sujname'])
            

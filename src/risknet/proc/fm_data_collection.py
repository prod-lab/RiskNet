import errno
import os
import threading
import urllib.request
import zipfile
from threading import Thread
import requests
import time
import yaml

# Constants
# example inputs: downloads 2009 Q1 and Q2 data and merges it into 2009 Q1 data files
login_page_url = 'https://freddiemac.embs.com/FLoan/secure/auth.php'
download_page_url = 'https://freddiemac.embs.com/FLoan/Data/download.php'


## gets the zipped files and unzips them
def get_data_from_url(file_name, fm_root):
    print(" STARTING   : " + str(file_name) + ".zip download ")
    download = urllib.request.urlretrieve('https://freddiemac.embs.com/FLoan/Data/' + str(file_name) + '.zip',
                                  fm_root + str(file_name) + '.zip')
    print(" SUCCESS   : " + str(file_name) + ".zip download complete ")
    try:
        print(" STARTING   : " + str(file_name) + ".zip unzip ")
        zip_ref = zipfile.ZipFile(fm_root + str(file_name) + '.zip', 'r')
        zip_ref.extractall(fm_root)
        zip_ref.close()
        print(" SUCCESS   : " + str(file_name) + ".zip unzip complete ")
        os.remove(fm_root + str(file_name) + '.zip')
        print(" SUCCESS   : " + str(file_name) + ".zip deletion complete ")
    except zipfile.BadZipfile:
        print(" ERROR   : Unsuccesfull to unzip "+fm_root + "/" + str(file_name) + '.zip')
        print (zipfile.BadZipfile)

#merges quarterly data together into q1 file
def append_txt_files(q1_path, files_to_append, fm_root):
    try:
        with open(fm_root + q1_path, 'a') as txt1_file:
            for txt_file_path in files_to_append:
                with open(fm_root + txt_file_path, 'r') as txt_file:
                    txt_content = txt_file.read()
                    txt1_file.write(txt_content)
                print("Contents of", txt_file_path, "appended to", q1_path)
                os.remove(fm_root + txt_file_path)
                print("Deleted: " + txt_file_path)
    except FileNotFoundError as e:
        print("One or more files not found:", e)

## Main Threading Execution
def start_execution(start_year, end_year, num_quarters, username, password, fm_root):
    with requests.Session() as sess:
        print(" SUCCESS   : Program Started Succesfully")
        sess.get(login_page_url);
        php_session_cookie = sess.cookies['PHPSESSID']
        login_payload = {'username' : username, 'password' : password,'cookie':php_session_cookie}
        response_login = sess.post(login_page_url, data = login_payload)
        download_page_payload = {'accept': 'Yes', 'action': 'acceptTandC', 'acceptSubmit': 'Continue', 'cookie': php_session_cookie}
        response_download = sess.post(download_page_url, data=download_page_payload)
        print( " SUCCESS   : Login into freddiemac.embs.com succesful ")

        print(" SUCCESS   : Threads creation started")
        threadspool=[]
        #Q1 file to append into
        years_orig = []
        years_perf = []
        #quarters to append to Q1
        years_orig_append =[]
        years_perf_append = []
        while end_year >= start_year :
            orig_append = []
            perf_append = []
            for i in range(num_quarters):
                quart_orig = 'historical_data_' + str(start_year) + 'Q' +  str(i+1)
                print(quart_orig)
                newThread=(Thread(target=get_data_from_url, args=(quart_orig,fm_root)))
                threadspool.append(newThread)
                quart_perf = 'historical_data_time_' + str(start_year) + 'Q' + str(i+1) + '.txt'
                quart_orig += '.txt' 
                #Make Q1 the file to append into
                if i == 0:       
                    years_orig.append(quart_orig)
                    years_perf.append(quart_perf)
                else:
                    orig_append.append(quart_orig)
                    perf_append.append(quart_perf)

            years_orig_append.append(orig_append)
            years_perf_append.append(perf_append)

            start_year += 1

        for eachThread in threadspool:
            eachThread.start()

        for eachThread in threadspool:
            eachThread.join()
        
        return years_orig, years_orig_append, years_perf, years_perf_append
        


# Run the freddie mac data collection process
# default inputs: downloads 2009 Q1 and Q2 data and merges it into 2009 Q1 data files
def run_data_collection(fm_root, start_year=2009, end_year=2009, num_quarters=2, merge_files=True):
    start = time.time()
    
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),'config','conf.yaml')
    with open(config_path) as conf:
        config = yaml.full_load(conf)
    
    fm_username = os.path.expanduser(config['data']['fm_username'])
    fm_password = os.path.expanduser(config['data']['fm_password'])

    years_orig, years_orig_append, years_perf, years_perf_append = start_execution(
        start_year, end_year, num_quarters, fm_username, fm_password, fm_root
        )
    #merge all data into Q1 file
    if merge_files:
        for i in range(len(years_orig)):
            append_txt_files(years_orig[i], years_orig_append[i], fm_root)
            append_txt_files(years_perf[i], years_perf_append[i], fm_root)
    
    end = time.time()

    #should take around 10 minutes with default args
    print(f" SUCCESS   : Freddy Mac download and quarter merges finished in {(end - start)/60} minutes")

if __name__ == "__main__":
    run_data_collection("src/risknet/data/")
from pydap.client import open_url
import requests
import os

# Specify 1980-01-01 Cloud OPeNDAP URL with T2M subsetting
dataset_url = 'https://opendap.earthdata.nasa.gov/collections/C1276812863-GES_DISC/granules/M2T1NXSLV.5.12.4%3AMERRA2_100.tavg1_2d_slv_Nx.19800101.nc4?dap4.ce=/T2M[0:1:23][0:1:360][0:1:575]'

# Set file path to root
token_file_path = os.path.join(os.path.expanduser("~"), ".edl_token")

# Read the token from the .edl_token file
with open(token_file_path, 'r') as token_file:
    token = token_file.read().strip()  # Ensure to strip any newlines or extra spaces

# Enter the token into the request header
my_session = requests.Session()
my_session.headers={"Authorization": token}

try:
    # Open the dataset using the request session containing your token
    dataset = open_url(dataset_url, session=my_session, protocol="dap4")

    # Stream and display the values of the 2-meter temperature variable
    print(dataset['T2M'][:])
except OSError as e:
    print('Error', e)
    print('Please check that your .edl_token file has been properly generated, and that your .dodsrc files are in their correct locations.')

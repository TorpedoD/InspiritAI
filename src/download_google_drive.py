from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import os
import io

# If modifying these SCOPES, delete the file token.json.
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

def authenticate_drive_api():
    """
    Authenticate and build the Google Drive API service.
    """
    creds = None
    credentials_file = 'credentials.json'

    # Check if 'credentials.json' exists, if not, provide instructions to create it
    if not os.path.exists(credentials_file):
        print(f"Error: '{credentials_file}' not found.")
        print("To create this file, follow these steps:")
        print("1. Go to the Google Cloud Console: https://console.developers.google.com/")
        print("2. Create a new project (or select an existing one).")
        print("3. Enable the Google Drive API for this project.")
        print("4. Go to 'Credentials' and create OAuth 2.0 Client IDs.")
        print("5. Download the credentials.json file and place it in this directory.")
        exit(1)

    # If 'token.json' exists, load it
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)

    # If there are no valid credentials, prompt the user to log in
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(credentials_file, SCOPES)
            creds = flow.run_local_server(port=0)
        
        # Save the credentials for the next run
        with open('token.json', 'w') as token:
            token.write(creds.to_json())

    return build('drive', 'v3', credentials=creds)

def download_files_from_folder(service, folder_id, download_dir):
    """
    Downloads all files from a specific Google Drive folder.

    Args:
    - service: Authenticated Google Drive API service instance.
    - folder_id: The ID of the Google Drive folder.
    - download_dir: Local directory where the files will be downloaded.
    """
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)

    # List all files in the folder
    results = service.files().list(
        q=f"'{folder_id}' in parents and trashed = false",
        fields="files(id, name, mimeType)"
    ).execute()

    items = results.get('files', [])

    if not items:
        print('No files found.')
    else:
        for item in items:
            file_id = item['id']
            file_name = item['name']
            print(f"Downloading {file_name}...")

            request = service.files().get_media(fileId=file_id)
            file_path = os.path.join(download_dir, file_name)

            with io.FileIO(file_path, 'wb') as fh:
                downloader = MediaIoBaseDownload(fh, request)
                done = False
                while not done:
                    status, done = downloader.next_chunk()
                    print(f"Download {file_name}: {int(status.progress() * 100)}%")

def main():
    # Authenticate and build the Drive API client
    service = authenticate_drive_api()

    # The ID of the folder you want to download files from
    folder_id = 'YOUR_GOOGLE_DRIVE_FOLDER_ID'  # Replace with your folder ID

    # The directory where you want to save the downloaded files
    download_dir = 'downloads'

    # Download the files
    download_files_from_folder(service, folder_id, download_dir)

if __name__ == '__main__':
    main()

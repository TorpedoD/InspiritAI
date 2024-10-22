

def main():
    # Authenticate and build the Drive API client
    service = authenticate_drive_api()

    # The ID of the folder you want to download files from
    folder_id = '1BxIrqbSvJsdBAzaQaVolzIvX017rttEq'  # Replace with your folder ID

    # The directory where you want to save the downloaded files
    download_dir = 'downloads'

    # Download the files
    download_files_from_folder(service, folder_id, download_dir)

if __name__ == '__main__':
  main()

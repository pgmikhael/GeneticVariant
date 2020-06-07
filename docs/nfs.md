## Setting up NFS (rbgquanta)
Adapted from: https://linuxize.com/post/how-to-install-and-configure-an-nfs-server-on-ubuntu-18-04

### Server IPs
The computer where directory located is called the server and computers or devices connecting to that server are called clients.

Obtain IP addresses with `hostname -I`:
- rbgquanta-storage1 (server) IP: 128.30.44.215
- rbgquanta1 (client) IP: 128.30.44.73
- rbgquanta2 (client) IP: 128.30.44.214


## Set up server:
To check the NFS server is not installed: `dpkg -l | grep nfs-kernel-server`

Install NFS package:
- `sudo apt update`
- `sudo apt install nfs-kernel-server`

When configuring an NFS server, it is a good practice is to use a global NFS root directory and bind mount the actual directories to the share mount point:
- `sudo mkdir -p /srv/nfs4/storage_nfs`
- `sudo mount --bind /storage/nfs /srv/nfs4/storage_nfs`
- Edit /etc/fstab:
    * Entry format: [Device] [Mount Point] [File System Type] [Options] [Dump] [Pass]
- `sudo nano /etc/fstab`
- `/storage/nfs /srv/nfs4/storage_nfs  none   bind   0   0`

Exporting the file systems:

The next step is to define the file systems that will be exported by the NFS server, the shares options and the clients that are allowed to access those file systems. Open the /etc/exports file:
- `sudo nano /etc/exports`
- Add the entries:
``` shell 
/srv/nfs4             128.30.44.73(rw,sync,no_subtree_check,crossmnt,fsid=0) 128.30.44.214(rw,sync,no_subtree_check,crossmnt,fsid=0)
/srv/nfs4/storage_nfs 128.30.44.73(rw,sync,no_subtree_check) 128.30.44.214(rw,sync,no_subtree_check)
```
- Save the file and export the shares: `sudo exportfs -ra`.  If there are any errors or warnings they will be shown on the terminal.


## Set up client:
- `sudo apt update`
- `sudo apt install nfs-common`

Mounting file systems:
- Create new directories for the mount points:`sudo mkdir -p /Mounts/rbg-storage1`

Mount the server: 
- `sudo mount -t nfs -o vers=4 128.30.44.215:/storage_nfs /Mounts/rbg-storage1`

Verify that the remote file systems are mounted successfully using either the mount or df command:
- `df -h`

To make the mounts permanent on reboot, edit the /etc/fstab file: 
- `sudo nano /etc/fstab`
- Add the entry:
``` shell 
128.30.44.215:/storage_nfs /Mounts/rbg-storage1   nfs   defaults,timeo=900,retrans=5,_netdev	0 0
```
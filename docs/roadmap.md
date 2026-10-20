# Roadmap and intended workflows

The intended end-to-end workflows are:

1. Extract a root filesystem from an OCI image and write its files into a
   valid ext4 image.
2. Copy or inject an application directory that lives on the host
   into a valid ext4 image.
3. Attach that image to a Linux VM, mount it as ext4, and run the application
   from the VM at native filesystem speed.
4. Format an ext4 fs so that it is optimized for AI-agents

The eventual CLI should support workflows conceptually like:

```text
ext4kit from-oci ghcr.io/example/service:latest service.ext4
ext4kit inject C:\work\my-app app-data.ext4 --destination /opt/my-app
```

The generated image can then be attached as a virtual disk and mounted by the
guest:

```sh
mount -t ext4 /dev/vdb /mnt/app
```

These workflows are still work in progress.

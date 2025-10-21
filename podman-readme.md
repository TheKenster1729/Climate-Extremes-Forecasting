# Climate Extremes Forecasting - Podman Container Guide

This guide explains how to build and run the Climate Extremes Forecasting application using Podman containers.

## Prerequisites

- Podman installed on your system
- Git repository cloned locally
- All required data files present in the repository

## Files Overview

- `ContainerFile` - Container build definition
- `extremes-2025.service` - Systemd service file for container management
- `.containerignore` - Files to exclude from container build

## Building the Container

```bash
# Build the container image
podman build -f ContainerFile -t extremes-2025 .

# Verify the image was created
podman images | grep extremes-2025
```

## Running the Container

### Method 1: Using Systemd Service (Recommended)

The systemd service file provides integration with automatic restarts and better service management.

#### Setup User Systemd Service

```bash
# Create user systemd directory
mkdir -p ~/.config/systemd/user

# Copy the service file
cp extremes-2025.service ~/.config/systemd/user/

# Reload systemd and start the service
systemctl --user daemon-reload
systemctl --user enable --now extremes-2025.service
```

#### Managing the Service

```bash
# Check service status
systemctl --user status extremes-2025.service

# View logs
journalctl --user -u extremes-2025.service -f

# Stop the service
systemctl --user stop extremes-2025.service

# Restart the service
systemctl --user restart extremes-2025.service

# Disable auto-start
systemctl --user disable extremes-2025.service
```

### Method 2: Manual Podman Run

```bash
# Run the container manually
podman run --replace \
  -v $(pwd):/home/extremes_user/Climate-Extremes-Forecasting \
  -p 8070:8050 \
  -d \
  --name extremes-2025 \
  extremes-2025
```

## Accessing the Application

Once the container is running, you can access the application at:

- **URL**: http://localhost:8070
- **Port**: 8070 (mapped from container port 8050)

## Container Details

### Image Information
- **Base Image**: Rocky Linux 8
- **Python Version**: 3.11
- **Virtual Environment**: `/home/extremes_user/venv`
- **Working Directory**: `/home/extremes_user/Climate-Extremes-Forecasting`
- **User**: `root`

### Volume Mounts
- Current directory mounted to `/home/extremes_user/Climate-Extremes-Forecasting`
- This allows the container to access all application files and data

### Port Mapping
- Host port 8070 → Container port 8050
- The application runs on port 8050 inside the container

## Troubleshooting

### Check Container Status

```bash
# List running containers
podman ps

# List all containers (including stopped)
podman ps -a

# Check container logs
podman logs extremes-2025
```

### Common Issues

1. **Port Already in Use**
   ```bash
   # Check what's using port 8070
   sudo netstat -tlnp | grep 8070
   
   # Stop the conflicting service or use a different port
   ```

2. **Container Won't Start**
   ```bash
   # Check container logs for errors
   podman logs extremes-2025
   
   # Run container interactively to debug
   podman run -it --rm extremes-2025 /bin/bash
   ```

3. **Permission Issues**
   ```bash
   # Ensure the container has access to the mounted directory
   ls -la /home/extremes_user/Climate-Extremes-Forecasting
   ```

### Cleanup

```bash
# Stop and remove the container
podman stop extremes-2025
podman rm extremes-2025

# Remove the image (optional)
podman rmi extremes-2025

# Stop and disable the systemd service
systemctl --user stop extremes-2025.service
systemctl --user disable extremes-2025.service
```

## Development

### Rebuilding After Changes

```bash
# Rebuild the container image
podman build -f ContainerFile -t extremes-2025 .

# Restart the service (if using systemd service)
systemctl --user restart extremes-2025.service
```

### Interactive Development

```bash
# Run container with interactive shell
podman run -it --rm \
  -v $(pwd):/home/extremes_user/Climate-Extremes-Forecasting \
  -p 8070:8050 \
  extremes-2025 /bin/bash
```

## Security Notes

- The container runs as root user for simplified setup
- Uses a Python virtual environment for dependency isolation
- Only necessary ports are exposed (8070)
- Volume mounts are read-write for development purposes

## Support

For issues related to:
- **Container setup**: Check this README and container logs
- **Application functionality**: Refer to the main application documentation
- **Data requirements**: Ensure all required CSV files are present in the repository
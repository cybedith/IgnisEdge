#!/bin/bash
# Reset P3 camera via /sys USB unbind/bind (no need to unplug cable)

VID="3474"
PID="45a2"

DEVICE=$(for d in /sys/bus/usb/devices/*/; do
    if [ -f "$d/idVendor" ] && [ -f "$d/idProduct" ]; then
        if [ "$(cat $d/idVendor 2>/dev/null)" = "$VID" ] && [ "$(cat $d/idProduct 2>/dev/null)" = "$PID" ]; then
            basename "$d"
            break
        fi
    fi
done)

if [ -z "$DEVICE" ]; then
    echo "P3 camera NOT found in USB devices."
    echo "Try: lsusb | grep 3474"
    exit 1
fi

echo "Found P3 at: $DEVICE"
echo "Unbinding..."
echo "$DEVICE" | sudo tee /sys/bus/usb/drivers/usb/unbind > /dev/null
sleep 2
echo "Rebinding..."
echo "$DEVICE" | sudo tee /sys/bus/usb/drivers/usb/bind > /dev/null
sleep 3
echo "Done. Camera should be ready."
lsusb | grep 3474

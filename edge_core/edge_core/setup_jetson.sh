#!/usr/bin/env bash
# ==============================================================================
# setup_jetson.sh -- Instalador y Bootstrap Automatizado para Jetson Orin Nano
# ==============================================================================
# Ejecutar directamente en la Jetson Orin Nano después del primer arranque:
#   chmod +x edge_core/setup_jetson.sh
#   ./edge_core/setup_jetson.sh
# ==============================================================================

set -e

GREEN="\033[92m"
RED="\033[91m"
YELLOW="\033[93m"
CYAN="\033[96m"
BOLD="\033[1m"
RESET="\033[0m"

echo -e "${BOLD}================================================================${RESET}"
echo -e "${BOLD}   IgnisEdge -- Instalador Automatizado en Jetson Orin Nano     ${RESET}"
echo -e "${BOLD}================================================================${RESET}"

# 1. Verificar Arquitectura ARM64
ARCH=$(uname -m)
echo -e "${CYAN}[1/6] Verificando arquitectura del sistema...${RESET}"
if [ "$ARCH" != "aarch64" ]; then
    echo -e "${YELLOW}[AVISO] No estás en aarch64 (detectado: $ARCH). Continuando de todos modos...${RESET}"
else
    echo -e "${GREEN}[OK] Arquitectura ARM64 (aarch64) verificada en Jetson.${RESET}"
fi

# 2. Configurar Regla udev para Cámara Térmica P3
echo -e "\n${CYAN}[2/6] Configurando permisos USB para cámara P3 (udev)...${RESET}"
UDEV_RULE_FILE="/etc/udev/rules.d/99-p3-camera.rules"
if [ ! -f "$UDEV_RULE_FILE" ]; then
    echo "Creando $UDEV_RULE_FILE..."
    sudo bash -c "cat > $UDEV_RULE_FILE" << 'UDEV_EOF'
SUBSYSTEM=="usb", ATTRS{idVendor}=="3474", ATTRS{idProduct}=="45a2", MODE="0666", GROUP="plugdev"
UDEV_EOF
    sudo udevadm control --reload-rules && sudo udevadm trigger
    sudo usermod -aG plugdev "$USER"
    echo -e "${GREEN}[OK] Regla udev configurada exitosamente.${RESET}"
else
    echo -e "${GREEN}[OK] Regla udev ya existente.${RESET}"
fi

# 3. Paquetes del Sistema
echo -e "\n${CYAN}[3/6] Verificando paquetes del sistema en Ubuntu...${RESET}"
sudo apt update -y
sudo apt install -y python3-pip python3-venv libusb-1.0-0-dev python3-opencv

# 4. Crear y Activar Entorno Virtual
IGNIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="$IGNIS_DIR/venv"
echo -e "\n${CYAN}[4/6] Configurando entorno virtual Python ($VENV_DIR)...${RESET}"

if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv --system-site-packages "$VENV_DIR"
    echo -e "${GREEN}[OK] venv creado con acceso a librerías del sistema (OpenCV JetPack).${RESET}"
else
    echo -e "${GREEN}[OK] venv existente detectado.${RESET}"
fi

source "$VENV_DIR/bin/activate"

# 5. Instalar Dependencias de Python
echo -e "\n${CYAN}[5/6] Instalando dependencias de producción en venv...${RESET}"
pip install --upgrade pip
pip install "numpy<2" scipy pandas scikit-learn joblib pyusb msgpack pymavlink pyserial tifffile imagecodecs

# Instalar driver de la P3 si existe la carpeta
if [ -d "$IGNIS_DIR/p3-ir-camera-main" ]; then
    echo "Instalando driver p3-ir-camera..."
    pip install -e "$IGNIS_DIR/p3-ir-camera-main"
fi

# 6. Ejecutar Batería de Pruebas de Validación
echo -e "\n${CYAN}[6/6] Ejecutando suite de pruebas unitarias de producción...${RESET}"
cd "$IGNIS_DIR"
python3 -m unittest discover -s edge_core/tests -p "test_*.py"

echo -e "\n${GREEN}${BOLD}================================================================${RESET}"
echo -e "${GREEN}${BOLD}   ¡INSTALACIÓN COMPLETADA AL 100%!                             ${RESET}"
echo -e "${GREEN}${BOLD}   Todos los módulos están listos para la Jetson Orin Nano.     ${RESET}"
echo -e "${GREEN}${BOLD}================================================================${RESET}"
echo -e "Próximos pasos:"
echo -e "  1. Diagnóstico de hardware:  python3 edge_core/tools/check_hardware.py"
echo -e "  2. Probar simulación:        python3 edge_core/launcher.py --sim"
echo -e "  3. Probar banco con P3:      python3 edge_core/launcher.py --bench"

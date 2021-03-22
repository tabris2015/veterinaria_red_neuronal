import random
from django.contrib.auth.models import User
from detection.models import Detection


def crear_admin():
    admin = User.objects.create_superuser('admin', 'admin@a.com', 'admin')


def run():
    print('creando admin...')
    crear_admin()

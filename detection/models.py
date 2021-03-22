from django.db import models


# Create your models here.
class Detection(models.Model):
    PERRO = 'dog'
    GATO = 'cat'
    ANIMAL = (
        (PERRO, 'Perro'),
        (GATO, 'Gato')
    )

    date = models.DateTimeField('detection date')
    animal = models.CharField(max_length=16, choices=ANIMAL)
    breed = models.CharField(max_length=128, null=True, blank=True)
    picture = models.ImageField(upload_to='pictures', blank=False, null=True, editable=False)
    rating = models.IntegerField(default=0)

    def __str__(self):
        return f'{self.animal} {self.breed}'

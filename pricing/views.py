from django.shortcuts import render
from django.http import HttpResponse
from django.template import loader

def index(request):
    return HttpResponse("Hello, world. You're at the pricing index.")

# Create your views here.

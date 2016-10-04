from django.shortcuts import render
from django.http import JsonResponse
from homepage.models import ContactMe
from homepage.src.goAgent import TestGo
from homepage.src.AgentHandler import AgentHandler

import random

# Create your views here.

def home(request):
    info = ContactMe.objects.all()

    return render(request, 'contact/contact.html',{'info': info})

def weiqi(request):
    return render(request, 'weiqi/index.html')

def nextMove(request):
    print("prepare next move")
    current_x = int(request.GET['x'])
    current_y = int(request.GET['y'])
    print(current_x,current_y)
    next_x, next_y = AgentHandler().next_move(current_x, current_y)
    ## the response need python integer instead of numpy.int
    move = {"row":int(next_y), "col":int(next_x)}
    return JsonResponse(move)

def agent(request):
    return render(request, 'weiqi/agent.html')

def agent_init(request):
    print("initiating")
    AgentHandler().init_agent()
    print("initiated")
    return JsonResponse({"code":1})



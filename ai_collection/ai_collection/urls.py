"""
URL configuration for ai_collection project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('fitur-1/', include('fitur-1.urls', namespace='fitur_1')),
    path('fitur-2/', include('fitur-2.urls', namespace='fitur_2')),
    path('fitur-3/', include('fitur-3.urls', namespace='fitur_3')),
    path('fitur-4/', include('fitur-4.urls', namespace='fitur_4')),
    path('fitur-5/', include('fitur-5.urls', namespace='fitur_5')),
    path('fitur-6/', include('fitur-6.urls', namespace='fitur_6')),
    path('fitur-7/', include('fitur-7.urls', namespace='fitur_7')),
    path('fitur-8/', include('fitur-8.urls', namespace='fitur_8')),
    path('fitur-9/', include('fitur-9.urls', namespace='fitur_9')),
    path('fitur-10/', include('fitur-10.urls', namespace='fitur_10')),
    path('fitur-11/', include('fitur-11.urls', namespace='fitur_11')),
    path('fitur-12/', include('fitur-12.urls', namespace='fitur_12')),
    path('fitur-13/', include('fitur-13.urls', namespace='fitur_13')),
    path('fitur-14/', include('fitur-14.urls', namespace='fitur_14')),
    path('fitur-15/', include('fitur-15.urls', namespace='fitur_15')),
    path('fitur-16/', include('fitur-16.urls', namespace='fitur_16')),
    path('fitur-17/', include('fitur-17.urls', namespace='fitur_17')),
    path('fitur-18/', include('fitur-18.urls', namespace='fitur_18')),
    path('fitur-19/', include('fitur-19.urls', namespace='fitur_19')),
    path('fitur-20/', include('fitur-20.urls', namespace='fitur_20')),
]

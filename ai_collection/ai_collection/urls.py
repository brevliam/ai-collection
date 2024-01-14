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
    # path('fitur-1/', include(('fitur_1.urls', 'fitur_1'), namespace='fitur_1')),
    # path('fitur-2/', include(('fitur_2.urls', 'fitur_2'), namespace='fitur_2')),
    # path('fitur-3/', include(('fitur_3.urls', 'fitur_3'), namespace='fitur_3')),
    # path('fitur-4/', include(('fitur_4.urls', 'fitur_4'), namespace='fitur_4')),
    # path('fitur-5/', include(('fitur_5.urls', 'fitur_5'), namespace='fitur_5')),
    # path('fitur-6/', include(('fitur_6.urls', 'fitur_6'), namespace='fitur_6')),
    # path('fitur-7/', include(('fitur_7.urls', 'fitur_7'), namespace='fitur_7')),
    # path('fitur-8/', include(('fitur_8.urls', 'fitur_8'), namespace='fitur_8')),
    # path('fitur-9/', include(('fitur_9.urls', 'fitur_9'), namespace='fitur_9')),
    # path('fitur-10/', include(('fitur_10.urls', 'fitur_10'), namespace='fitur_10')),
    # path('fitur-11/', include(('fitur_11.urls', 'fitur_11'), namespace='fitur_11')),
    # path('fitur-12/', include(('fitur_12.urls', 'fitur_12'), namespace='fitur_12')),
    # path('fitur-13/', include(('fitur_13.urls', 'fitur_13'), namespace='fitur_13')),
    # path('fitur-14/', include(('fitur_14.urls', 'fitur_14'), namespace='fitur_14')),
    # path('fitur-15/', include(('fitur_15.urls', 'fitur_15'), namespace='fitur_15')),
    path('fitur-16/', include(('fitur_16.urls', 'fitur_16'), namespace='fitur_16')),
    # path('fitur-17/', include(('fitur_17.urls', 'fitur_17'), namespace='fitur_17')),
    # path('fitur-18/', include(('fitur_18.urls', 'fitur_18'), namespace='fitur_18')),
    # path('fitur-19/', include(('fitur_19.urls', 'fitur_19'), namespace='fitur_19')),
    # path('fitur-20/', include(('fitur_20.urls', 'fitur_20'), namespace='fitur_20')),
]

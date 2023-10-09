from django.urls import path
from .views import DebtorLabelByAge, DebtorLabelByLocation, DebtorLabelByBehavior, DebtorLabelByCharacter
from .views import DebtorLabelByCollectorField, DebtorLabelBySES, DebtorLabelByDemography
from .views import CollectorLabelByAge,  CollectorLabelByLocation,  CollectorLabelByBehavior,  CollectorLabelByCharacter
from .views import CollectorLabelByCollectorField,  CollectorLabelBySES,  CollectorLabelByDemography


urlpatterns = [
    path('debtorlabelage/', DebtorLabelByAge.as_view(), name='debtor_age_label'),
    path('debtorlabellocation/', DebtorLabelByLocation.as_view(), name='debtor_location_label'),
    path('debtorlabelbehavior/', DebtorLabelByBehavior.as_view(), name='debtor_behavior_label'),
    path('debtorlabelcharacter/', DebtorLabelByCharacter.as_view(), name='debtor_character_label'),
    path('debtorlabelcollectorfield/', DebtorLabelByCollectorField.as_view(), name='debtor_collector field_label'),
    path('debtorlabelses/', DebtorLabelBySES.as_view(), name='debtor_ses_label'),
    path('debtorlabeldemography/', DebtorLabelByDemography.as_view(), name='debtor_demography_label'),
    path('collectorlabelage/', CollectorLabelByAge.as_view(), name='collector_age_label'),
    path('collectorlabellocation/', CollectorLabelByLocation.as_view(), name='collector_location_label'),
    path('collectorlabelbehavior/', CollectorLabelByBehavior.as_view(), name='collector_behavior_label'),
    path('collectorlabelcharacter/', CollectorLabelByCharacter.as_view(), name='collector_character_label'),
    path('collectorlabelcollectorfield/', CollectorLabelByCollectorField.as_view(), name='collector_collector field_label'),
    path('collectorlabelses/', CollectorLabelBySES.as_view(), name='collector_ses_label'),
    path('collectorlabeldemography/', CollectorLabelByDemography.as_view(), name='collector_demography_label'),

]

FIT_DONE = False
SETUP_DONE = False
LOAD_DONE = False
COMPILE_DONE = False

model = None

class_names_expected = ['Colon','Heart','Blood','Vagina','Thyroid','Liver','Salivary_Gland',
                        'Pancreas','Cervix_Uteri','Prostate','Ovary','Skin','Pituitary',
                        'Small_Intestine','Fallopian_Tube','Adrenal_Gland','Nerve',
                        'Adipose_Tissue','Spleen','Stomach','Muscle','Blood_Vessel','Lung',
                        'Esophagus','Brain','Testis','Uterus','Kidney','Bladder','Breast']

ds = None
info = None
X = None
y = None
class_names = None
x_train = None
y_train = None
x_validation = None
y_validation = None
x_test = None
y_test = None
history = None

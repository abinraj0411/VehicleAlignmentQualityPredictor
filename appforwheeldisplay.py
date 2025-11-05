import pygame
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# ------------------------- MODEL TRAINING ----------------------------
df = pd.read_csv("vehicle_alignment_quality_dataset.csv")
X = df[["Camber (°)", "Caster (°)", "Toe (mm)", "Vehicle Age (years)"]]
y = df["Alignment Quality"]

le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
joblib.dump(model, "alignment_model.pkl")

# --------------------------- PYGAME GUI -------------------------------
pygame.init()
WIDTH, HEIGHT = 800, 550
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Vehicle Alignment Quality Predictor")

font = pygame.font.SysFont("arial", 18, bold=True)
small_font = pygame.font.SysFont("arial", 14)
big_font = pygame.font.SysFont("arial", 20, bold=True)

bg_color = (245, 245, 245)
text_color = (20, 20, 20)
input_bg = (255, 255, 255)
hover_bg = (230, 240, 255)
button_color = (50, 100, 180)
button_hover = (70, 130, 230)
green, yellow, red = (0, 200, 0), (220, 180, 0), (200, 0, 0)

# Load image
wheel_image = pygame.image.load("Screenshot-2025-11-04 001411.png")
wheel_image = pygame.transform.scale(wheel_image, (300, 250))

# Input Box Class
class InputBox:
    def __init__(self, x, y, w, h, placeholder):
        self.rect = pygame.Rect(x, y, w, h)
        self.color = input_bg
        self.text = ""
        self.placeholder = placeholder
        self.active = False

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            self.active = self.rect.collidepoint(event.pos)
        if event.type == pygame.KEYDOWN and self.active:
            if event.key == pygame.K_RETURN:
                self.active = False
            elif event.key == pygame.K_BACKSPACE:
                self.text = self.text[:-1]
            else:
                self.text += event.unicode

    def draw(self, screen):
        pygame.draw.rect(screen, hover_bg if self.active else self.color, self.rect, border_radius=6)
        txt_surface = small_font.render(self.text if self.text else self.placeholder, True, (0, 0, 0))
        screen.blit(txt_surface, (self.rect.x + 5, self.rect.y + 6))

    def get_value(self):
        try:
            return float(self.text)
        except:
            return None

# Button
def draw_button(text, x, y, w, h, hover):
    color = button_hover if hover else button_color
    pygame.draw.rect(screen, color, (x, y, w, h), border_radius=8)
    label = big_font.render(text, True, (255, 255, 255))
    screen.blit(label, (x + (w - label.get_width())/2, y + (h - label.get_height())/2))

# Predict Function
def predict_quality(camber, caster, toe, age):
    if None in [camber, caster, toe, age]:
        return "Invalid"
    pred = model.predict([[camber, caster, toe, age]])
    return le.inverse_transform(pred)[0]

# Input Boxes
left_inputs = [
    InputBox(40, 60, 160, 30, "Camber (°)"),
    InputBox(40, 100, 160, 30, "Caster (°)"),
    InputBox(40, 140, 160, 30, "Toe (mm)"),
]
right_inputs = [
    InputBox(600, 60, 160, 30, "Camber (°)"),
    InputBox(600, 100, 160, 30, "Caster (°)"),
    InputBox(600, 140, 160, 30, "Toe (mm)"),
]
age_input = InputBox(320, 330, 160, 30, "Vehicle Age (years)")

# Main loop
running = True
left_result = ""
right_result = ""
clock = pygame.time.Clock()

while running:
    screen.fill(bg_color)
    mouse_pos = pygame.mouse.get_pos()

    # Title
    title = font.render("Vehicle Alignment Quality Predictor", True, text_color)
    screen.blit(title, (20, 20))

    # Draw inputs
    for box in left_inputs + right_inputs + [age_input]:
        box.draw(screen)

    # Draw middle image
    screen.blit(wheel_image, (WIDTH//2 - 150, 70))

    # Draw labels
    screen.blit(font.render("Left", True, text_color), (100, 35))
    screen.blit(font.render("Right", True, text_color), (660, 35))
    screen.blit(small_font.render(f"Accuracy: {accuracy*100:.1f}%", True, (80, 80, 80)), (WIDTH//2 - 50, 370))

    # Predict Button
    button_rect = pygame.Rect(WIDTH//2 - 70, 420, 140, 40)
    hover = button_rect.collidepoint(mouse_pos)
    draw_button("Predict", button_rect.x, button_rect.y, button_rect.w, button_rect.h, hover)

    # Display results
    if left_result:
        color = green if left_result == "Good" else yellow if left_result == "Average" else red
        screen.blit(font.render(left_result, True, color), (90, 200))
    if right_result:
        color = green if right_result == "Good" else yellow if right_result == "Average" else red
        screen.blit(font.render(right_result, True, color), (650, 200))

    # Events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        for box in left_inputs + right_inputs + [age_input]:
            box.handle_event(event)
        if event.type == pygame.MOUSEBUTTONDOWN and hover:
            left_values = [b.get_value() for b in left_inputs] + [age_input.get_value()]
            right_values = [b.get_value() for b in right_inputs] + [age_input.get_value()]
            left_result = predict_quality(*left_values)
            right_result = predict_quality(*right_values)

    pygame.display.flip()
    clock.tick(30)

pygame.quit()

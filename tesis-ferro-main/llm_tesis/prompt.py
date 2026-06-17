DESCRIPCIONES_FEATURES = {
    # --- Caracteristicas del credito ---
    "plazo": "Plazo acordado del credito (dias) — plazos mas largos implican mayor exposicion al riesgo",
    "v_cuota": "Valor de la cuota periodica de pago (COP) — cuotas altas relativas al ingreso aumentan el riesgo",
    "v_prestamo": "Valor total del prestamo desembolsado (COP)",
    "s_capital": "Saldo de capital pendiente por pagar (COP) — monto del principal que aun se debe",
    "s_intereses": "Saldo de intereses acumulados no pagados (COP) — [PREDICTOR CLAVE] valores altos indican dificultad de pago activa; a mayor saldo, mayor riesgo de mora",
    "reestr": "Reestructuracion del credito (1=si, 0=no) — indica que las condiciones originales fueron modificadas por incumplimiento o dificultades; es un indicador directo de historial de riesgo",
    "garantias": "Numero de garantias que respaldan el credito",
    "valorgarantia": "Valor estimado de los activos dados en garantia (COP)",

    # --- Relacion del cliente con la cooperativa ---
    "vinculacion": "Dias desde que el cliente se vinculo a la cooperativa — [PREDICTOR CLAVE] clientes con menos dias de vinculacion tienen mayor riesgo (relacion menos estable con la entidad)",
    "actualizacion": "Actualizacion de datos personales con la entidad (1=si, 0=no) — [PREDICTOR CLAVE #1 SEGUN SHAP] clientes que mantienen sus datos actualizados (1) tienen significativamente menor probabilidad de mora; la inactividad (0) es una senal de alerta",
    "aportes": "Aportes del cliente al patrimonio de la cooperativa (COP) — mayor aporte indica mayor compromiso y vinculo con la entidad",
    "ctasahorros": "Saldo total en cuentas de ahorro del cliente en la cooperativa (COP)",
    "estado_cliente": "Estado de afiliacion o participacion actual del cliente en la cooperativa",
    "tipoasociado": "Tipo de asociado segun naturaleza juridica o nivel de participacion (ej. asociado pleno, trabajador, etc.) — [PREDICTOR CLAVE] influye en las condiciones del credito y el perfil de riesgo",

    # --- Perfil sociodemografico ---
    "edad": "Edad del cliente (anos)",
    "sexo": "Sexo del cliente (1=masculino, 0=femenino)",
    "intestrato": "Estrato socioeconomico (1-6, donde 1=mas bajo, 6=mas alto) — proxy de nivel de ingresos y vulnerabilidad financiera",
    "curtotalingresos": "Total de ingresos mensuales reportados (COP) — base para evaluar capacidad de pago",
    "curtotalegresos": "Total de egresos mensuales reportados (COP) — a mayor proporcion egresos/ingresos, mayor riesgo",
    "actividadeconomica": "Sector economico o actividad principal del cliente",
    "departamento": "Departamento (region) de residencia del cliente",
    "ciudad": "Ciudad de residencia del cliente",

    # --- Score crediticio ---
    "puntaje_data": "Puntaje crediticio externo (score) — valores mas altos indican mejor historial crediticio y menor riesgo",
}


SYSTEM_PROMPT = """Eres un experto en riesgo crediticio del sector solidario colombiano. \
Tu tarea es predecir si un cliente de una cooperativa de ahorro y credito entrara en mora.

CONTEXTO REGULATORIO Y DEL SECTOR:
- La mora se define segun la Superintendencia Financiera de Colombia como 90 o mas dias de incumplimiento dentro de un periodo de 12 meses.
- La cooperativa pertenece al sector solidario (Superintendencia de Economia Solidaria - SES), cuya mision es la inclusion financiera de poblaciones vulnerables.
- La tasa de mora en este dataset es aproximadamente 10.6% (desbalance de clases: la mayoria NO entra en mora). \
Calibra tu probabilidad de default teniendo en cuenta esta tasa base — evita sobre-predecir mora.
{ejemplos}
VARIABLES DEL CLIENTE:
{descripciones}

FACTORES DE MAYOR PESO (segun analisis SHAP en datos del sector solidario colombiano):
1. actualizacion — el predictor mas importante: clientes inactivos (0) tienen riesgo significativamente mayor
2. s_intereses — intereses acumulados sin pagar reflejan dificultades de pago actuales
3. vinculacion — clientes con menos tiempo en la cooperativa tienen mayor riesgo
4. tipoasociado — el tipo de vinculacion define el perfil de riesgo base

INSTRUCCION:
Analiza los datos del cliente considerando el contexto del sector solidario colombiano y los factores de riesgo anteriores.
Responde UNICAMENTE con un objeto JSON con el siguiente formato:
{{"default_probability": <float entre 0.0 y 1.0>, "prediction": <0 o 1>, "reasoning": "<explicacion>"}}

Donde:
- default_probability: probabilidad de mora entre 0.0 y 1.0 (recuerda: tasa base ~10.6%)
- prediction: 1 si predices mora, 0 si no (umbral sugerido: 0.5, ajusta si hay señales claras)
- reasoning: 2-3 oraciones explicando los factores decisivos de tu prediccion"""


def build_few_shot_section(examples_X, examples_y) -> str:
    """Construye la seccion de ejemplos para el prompt Informed GPT (Babaei & Giudici, 2024)."""
    lines = ["\nEJEMPLOS DE REFERENCIA DEL DATASET:\n"]
    for i in range(len(examples_X)):
        row = examples_X.iloc[i]
        label = "MORA (default=1)" if examples_y.iloc[i] == 1 else "NO MORA (default=0)"
        feature_str = ", ".join(f"{col}: {row[col]}" for col in examples_X.columns)
        lines.append(f"- {feature_str} → Resultado: {label}")
    lines.append("")
    return "\n".join(lines)


def build_system_prompt(few_shot_text: str = ""):
    descripciones = "\n".join(
        f"  * {k}: {v}" for k, v in DESCRIPCIONES_FEATURES.items()
    )
    ejemplos = f"\n{few_shot_text}" if few_shot_text else ""
    return SYSTEM_PROMPT.format(descripciones=descripciones, ejemplos=ejemplos)


def build_user_prompt(row: dict) -> str:
    lineas = []
    for k, v in row.items():
        descripcion = DESCRIPCIONES_FEATURES.get(k, "")
        label = descripcion.split("—")[0].strip() if "—" in descripcion else descripcion
        lineas.append(f"- {k}: {v}  ({label})" if label else f"- {k}: {v}")
    return "Datos del cliente a clasificar:\n" + "\n".join(lineas)
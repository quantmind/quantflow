{{- define "service.app" }}
{{ $root := . }}
{{ $values := pluck .name .Values | first }}
{{ $dash_name := .name | replace "_" "-" }}
{{ $image :=  default .Values.image $values.image }}
{{ $livenessProbe := default .Values.livenessProbe $values.livenessProbe }}
{{ $readinessProbe := default .Values.readinessProbe $values.readinessProbe }}
{{- $service_name := printf "%s-%s" (include "quantflow.name" $root) $dash_name -}}
apiVersion: v1
kind: Service
metadata:
  name: {{ $service_name }}
  labels:
    app: {{ $service_name }}
    metrics: {{ template "quantflow.name" . }}-metrics
    chart: {{ template "quantflow.chart" . }}
    release: {{ .Release.Name }}
    heritage: {{ .Release.Service }}
spec:
  type: {{ $values.type }}
  ports:
    - name: http
      port: 80
      targetPort: {{ .Values.targetPort }}
      protocol: TCP
  selector:
    app: {{ $service_name }}
    release: {{ .Release.Name }}
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {{ $service_name }}
  labels:
    app: {{ $service_name }}
    chart: {{ template "quantflow.chart" . }}
    release: {{ .Release.Name }}
    heritage: {{ .Release.Service }}
spec:
  replicas: {{ $values.replicaCount }}
  selector:
    matchLabels:
      app: {{ $service_name }}
      release: {{ .Release.Name }}
  template:
    metadata:
      labels:
        app: {{ $service_name }}
        release: {{ .Release.Name }}
      {{- with $values.podAnnotations }}
      annotations:
        {{- toYaml . | nindent 8 }}
      {{- end }}
    spec:
      containers:
        - name: main
          image: "{{ default .Values.image.registry $image.registry }}/{{ default .Values.image.repository $image.repository }}:{{ .Values.image.tag }}"
          imagePullPolicy: {{ .Values.image.pullPolicy }}
          ports:
            - name: http
              containerPort: {{ .Values.targetPort }}
              protocol: TCP
          livenessProbe:
            {{- toYaml $livenessProbe | nindent 12 }}
          readinessProbe:
            {{- toYaml $readinessProbe | nindent 12 }}
          command:
            {{- toYaml $values.command | nindent 12 }}
          resources:
            requests:
              memory: {{ $values.requests.memory }}
              cpu: {{ $values.requests.cpu }}
            limits:
              memory: {{ $values.limits.memory }}
              cpu: {{ $values.limits.cpu }}
          envFrom:
            - configMapRef:
                name: {{ include "quantflow.name" $root }}
            - secretRef:
                name: {{ include "quantflow.name" $root }}
          env:
          - name: MICRO_SERVICE_PORT
            value: "{{ .Values.targetPort }}"
          - name: APP_NAME
            value: "quantflow-{{ $dash_name }}"
          {{- range $key, $value := $values.vars }}
          - name: {{ $key }}
            value: {{ $value | quote }}
          {{- end }}
{{- end }}

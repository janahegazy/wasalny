FROM nginx:alpine
COPY ["waslny_cairo_realmap (1).html", "/usr/share/nginx/html/index.html"]
EXPOSE 80
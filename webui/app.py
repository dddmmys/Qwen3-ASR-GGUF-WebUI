from flask import Flask
from routes import register_routes

app = Flask(__name__)

# 注册所有路由
register_routes(app)

if __name__ == "__main__":
    # 注意：debug=True 会输出更多信息，且代码修改后自动重启
    app.run(host="127.0.0.1", port=5000, debug=True)

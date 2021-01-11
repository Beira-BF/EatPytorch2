# python之Argparse模块
# argparse模块可以轻松编写用户友好的命令行接口。程序定义它需要的参数，然后argparse将弄清楚如何从sys.argv解析出那些参数。
# argparse模块还会自动生成帮助和使用手册，并在用户给程序传入无效参数时报出错误信息。

# argparse简单使用流程
# 主要有三个步骤：
# 创建ArgumentParser()对象
# 调用add_argument()方法添加参数
# 使用parse_args()解析添加的参数

# 创建解析器对象

import argparse
parser = argparse.ArgumentParser()
# parser.add_argument("echo", help="echo the string you use here")
# args = parser.parse_args()
#
# print(args.echo)
# 添加可选参数声明的参数名前缀带-或--,前缀是-的为短参数，前缀是--是长参数，两者可以都有，也可以只有一个,短参数和长参数效果一样。
# 可选参数的值接在位置参数的后面，不影响位置参数的解析顺序。
# 以深度学习训练中经常出现的为例：

parser.add_argument("--verbosity", help="increase output verbosity", action="store_true")
parser.add_argument("square", help="display a square of a given number", type=int)
parser.add_argument("cube", help="display a cube of a given number", type=int)
parser.add_argument("--batch-size", type=int, default=64, metavar='N', help="input batch size for training (default:64)")
parser.add_argument("--save_model", action="store_true", default=False, help="For saving the current Model")
parser.add_argument("-a", help="input a int")

# 其中action参数的'store_true'指的是：触发action时为真，不触发则为假。即储存了一个bool变量，默认为false，触发不用赋值即变为true
# type:
# 指定参数类别，默认是str，传入数字要定义
# help：是一些提示信息
# default：是默认值
# metavar: 在 usage 说明中的参数名称，对于必选参数默认就是参数名称，对于可选参数默认是全大写的参数名称.
# 其它详细用法文档介绍：https://docs.python.org/zh-cn/3/library/argparse.html#argparse.ArgumentParser.add_argument


args = parser.parse_args()
if args.verbosity:
    print("verbosity turned on")

print(args.square**2)
print(args.cube**3)
print(args.a)


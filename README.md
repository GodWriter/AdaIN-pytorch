# AdaIN-pytorch
Arbitrary Style Transfer in Pytorch



> Training Command

```bash
python main.py --module train --content_dir yours --style_dir yours
```



> Testing Command

```bash
python main.py --module test --content yours --style yours --alpha 0.8
```



> Result Show

| content image                                                | style image                                                  | result                                                       |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| <img src="result/content.jpg" alt="content" style="zoom: 45%;" /> | <img src="result/style.png" alt="style" style="zoom: 75%;" /> | <img src="result/output.jpg" alt="result" style="zoom: 45%;" /> |


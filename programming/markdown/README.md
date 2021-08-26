# Markdown

## Markdown Editor

We will use  'Typora' as the offline Markdown editor. 

Typora:  [Download from here](https://typora.io/#windows)

## What is Markdown?

[Markdown](http://daringfireball.net/projects/markdown/) is a way to style text on the web. You control the display of the document; formatting words as bold or italic, adding images, and creating lists are just a few of the things we can do with Markdown. Mostly, Markdown is just regular text with a few non-alphabetic characters thrown in, like `#` or `*`.

You can use Markdown most places around GitHub:

* [Gists](https://gist.github.com/)
* Comments in Issues and Pull Requests
* Files with the `.md` or `.markdown` extension:  README.md

## Syntax guide

Here’s an overview of Markdown syntax that you can use anywhere on GitHub.com or in your own text files.

#### Headers

```text
# This is an <h1> tag
## This is an <h2> tag
###### This is an <h6> tag
```

#### Emphasis

```text
*This text will be italic*
_This will also be italic_

**This text will be bold**
__This will also be bold__

_You **can** combine them_
```

**Unordered List**

```text
* Item 1
* Item 2
  * Item 2a
  * Item 2b
```

**Ordered List**

```text
1. Item 1
1. Item 2
1. Item 3
   1. Item 3a
   1. Item 3b
```

#### Images

```text
Format: ![Alt Text](url)

![GitHub Logo](/images/logo.png)
![Image of Yaktocat](https://octodex.github.com/images/yaktocat.png)
```

#### Links

```text
http://github.com - automatic!
[GitHub](http://github.com)
```

#### Blockquotes

```text
As Kanye West said:

> We're living the future so
> the present is our past.
```

#### Inline code

```text
I think you should use an
`<addr>` element here instead.
```

**Block code**

```text
```javascript
function fancyAlert(arg) {
  if(arg) {
    $.facebox({div:'#foo'})
  }
}
```


```C
int main()
{
 return 0;
}
```
```



## Examples

* [Text](https://guides.github.com/features/mastering-markdown/#)
* [Lists](https://guides.github.com/features/mastering-markdown/#)
* [Images](https://guides.github.com/features/mastering-markdown/#)
* [Headers & Quotes](https://guides.github.com/features/mastering-markdown/#)
* [Code](https://guides.github.com/features/mastering-markdown/#)
* [Extras](https://guides.github.com/features/mastering-markdown/#)

```text
There are many different ways to style code with GitHub's markdown. If you have inline code blocks, wrap them in backticks: `var example = true`.  If you've got a longer block of code, you can indent with four spaces:

    if (isAwesome){
      return true
    }

GitHub also supports something called code fencing, which allows for multiple lines without indentation:

```
if (isAwesome){
  return true
}
```

And if you'd like to use syntax highlighting, include the language:

```javascript
if (isAwesome){
  return true
}
```
```

There are many different ways to style code with GitHub’s markdown. If you have inline code blocks, wrap them in backticks: `var example = true`. If you’ve got a longer block of code, you can indent with four spaces:

```text
if (isAwesome){
  return true
}
```

GitHub also supports something called code fencing, which allows for multiple lines without indentation:

```text
if (isAwesome){
  return true
}
```

And if you’d like to use syntax highlighting, include the language:

```text
if (isAwesome){
  return true
}
```

## Reference

{% embed url="https://guides.github.com/features/mastering-markdown/\#syntax" %}

## 


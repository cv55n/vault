#ifndef DLLISTDELEGATE_H
#define DLLISTDELEGATE_H

#include <QAbstractItemDelegate>
#include <QModelIndex>
#include <QPainter>
#include <QStyleOptionProgressBarV2>
#include <QProgressBar>
#include <QApplication>
#include "misc.h"

// define para colunas de lista de downloads
#define NAME 0
#define SIZE 1
#define PROGRESS 2
#define DLSPEED 3
#define UPSPEED 4
#define STATUS 5
#define ETA 6

class DLListDelegate: public QAbstractItemDelegate {
    Q_OBJECT

    public:
        DLListDelegate(QObject *parent=0) : QAbstractItemDelegate(parent){}
        ~DLListDelegate(){}

        void paint(QPainter * painter, const QStyleOptionViewItem & option, const QModelIndex & index) const{
            QStyleOptionViewItem opt = option;

            char tmp[MAX_CHAR_TMP];

            // determina a cor do texto
            QVariant value = index.data(Qt::TextColorRole);

            if (value.isValid() && qvariant_cast<QColor>(value).isValid()){
                opt.palette.setColor(QPalette::Text, qvariant_cast<QColor>(value));
            }

            QPalette::ColorGroup cg = option.state & QStyle::State_Enabled
                                    ? QPalette::Normal : QPalette::Disabled;

            if (option.state & QStyle::State_Selected){
                painter->setPen(opt.palette.color(cg, QPalette::HighlightedText));
            }else{
                painter->setPen(opt.palette.color(cg, QPalette::Text));
            }

            // desenha a cor do fundo
            if(index.column() != PROGRESS){
                if (option.showDecorationSelected && (option.state & QStyle::State_Selected)){
                    if (cg == QPalette::Normal && !(option.state & QStyle::State_Active)){
                        cg = QPalette::Inactive;
                    }

                    painter->fillRect(option.rect, option.palette.brush(cg, QPalette::Highlight));
                }else{
                    value = index.data(Qt::BackgroundColorRole);

                    if (value.isValid() && qvariant_cast<QColor>(value).isValid()){
                        painter->fillRect(option.rect, qvariant_cast<QColor>(value));
                    }
                }
            }

            switch(index.column()){
                case SIZE:
                    painter->drawText(option.rect, Qt::AlignCenter, misc::friendlyUnit(index.data().toLongLong()));
                    break;
                case ETA:
                    painter->drawText(option.rect, Qt::AlignCenter, misc::userFriendlyDuration(index.data().toLongLong()));
                    break;
                case UPSPEED:
                case DLSPEED:{
                    float speed = index.data().toDouble();

                    snprintf(tmp, MAX_CHAR_TMP, "%.1f", speed/1024.);

                    painter->drawText(option.rect, Qt::AlignCenter, QString(tmp)+" "+tr("KiB/s"));
                    
                    break;
                }
                case PROGRESS:{
                    QStyleOptionProgressBarV2 newopt;

                    float progress;

                    progress = index.data().toDouble()*100.;

                    snprintf(tmp, MAX_CHAR_TMP, "%.1f", progress);

                    newopt.rect = opt.rect;
                    newopt.text = QString(tmp)+"%";
                    newopt.progress = (int)progress;
                    newopt.maximum = 100;
                    newopt.minimum = 0;
                    newopt.state |= QStyle::State_Enabled;
                    newopt.textVisible = false;

                    QApplication::style()->drawControl(QStyle::CE_ProgressBar, &newopt, painter);
                    
                    // preferimos exibir o texto manualmente para controlar a cor/fonte/negrito
                    if (option.state & QStyle::State_Selected){
                        opt.palette.setColor(QPalette::Text, QColor("grey"));

                        painter->setPen(opt.palette.color(cg, QPalette::Text));
                    }

                    painter->drawText(option.rect, Qt::AlignCenter, newopt.text);

                    break;
                }
                case NAME:{
                    // decoração
                    value = index.data(Qt::DecorationRole);
                    
                    QPixmap pixmap = qvariant_cast<QIcon>(value).pixmap(option.decorationSize, option.state & QStyle::State_Enabled ? QIcon::Normal : QIcon::Disabled, option.state & QStyle::State_Open ? QIcon::On : QIcon::Off);
                    QRect pixmapRect = (pixmap.isNull() ? QRect(0, 0, 0, 0): QRect(QPoint(0, 0), option.decorationSize));
                    
                    if (pixmapRect.isValid()){
                        QPoint p = QStyle::alignedRect(option.direction, Qt::AlignLeft, pixmap.size(), option.rect).topLeft();
                        
                        painter->drawPixmap(p, pixmap);
                    }

                    painter->drawText(option.rect.translated(pixmap.size().width(), 0), Qt::AlignLeft, index.data().toString());
                    
                    break;
                }

                default:
                    painter->drawText(option.rect, Qt::AlignCenter, index.data().toString());
            }
        }

        QSize sizeHint(const QStyleOptionViewItem & option, const QModelIndex & index) const{
            QVariant value = index.data(Qt::FontRole);
            QFont fnt = value.isValid() ? qvariant_cast<QFont>(value) : option.font;
            QFontMetrics fontMetrics(fnt);
            
            const QString text = index.data(Qt::DisplayRole).toString();
            
            QRect textRect = QRect(0, 0, 0, fontMetrics.lineSpacing() * (text.count(QLatin1Char('\n')) + 1));
            
            return textRect.size();
        }
};

#endif